#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cblas.h>
#include"istalib.h"

ISTAinstance* ISTAinstance_new(float* A, int ldA, int rdA, float* b, float lambda, float gamma, 
			       int acceleration, char regressionType, float* xvalue, float step )
{
  // This method initializes an ISTAinstance object
  ISTAinstance* instance = malloc(sizeof(ISTAinstance));
  if ( instance==NULL )
    printf("Unable to allocate memory\n");

  instance->A = A;
  instance->ldA = ldA;
  instance->rdA = rdA;
  instance->b = b;
  instance->lambda = lambda;
  instance->gamma = gamma;
  instance->acceleration = acceleration;
  instance->regressionType = regressionType;
  instance->xcurrent = xvalue;

  //Allocate memory for values used during the calculation
  instance->stepsize = malloc(sizeof(float));
  *(instance->stepsize) = step;

  instance->xprevious = malloc(rdA*sizeof(float));
  if ( instance->xprevious==NULL )
    printf("Unable to allocate memory\n");

  instance->searchPoint = malloc(rdA*sizeof(float));
  if ( instance->searchPoint==NULL )
    printf("Unable to allocate memory\n");
  cblas_scopy(rdA, instance->xcurrent, 1, instance->searchPoint, 1);

  instance->gradvalue = malloc(rdA*sizeof(float));
  instance->eta = malloc((ldA+rdA)*sizeof(float));
  if ( instance->gradvalue==NULL || instance->eta==NULL )
    printf("Unable to allocate memory\n");

  return instance;
}


void ISTAinstance_free(ISTAinstance* instance)
{
  // Frees an entire ISTAinstance pointer
  free(instance->eta); 
  free(instance->gradvalue); 
  free(instance->searchPoint); 
  free(instance->xprevious);
  free(instance->xcurrent);
  free(instance->stepsize); 
  free(instance-> b);
  free(instance-> A);
  free(instance);
}



void ISTAsolve(float* A, int ldA, int rdA, float* b, float lambda, float gamma, 
	       int acceleration, char regressionType, float* xvalue, 
	       int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF)
{
  //This method updates xvalue to reflect the solution to the optimization

  //INITIALIZATION
  ISTAinstance* instance = malloc(sizeof(ISTAinstance));
  if ( instance==NULL )
    printf("Unable to allocate memory\n");

  instance->A = A;
  instance->ldA = ldA;
  instance->rdA = rdA;
  instance->b = b;
  instance->lambda = lambda;
  instance->gamma = gamma;
  instance->acceleration = acceleration;
  instance->regressionType = regressionType;
  instance->xcurrent = xvalue;

  //Allocate memory for values used during the calculation
  instance->stepsize = malloc(sizeof(float));
  *(instance->stepsize) = 1.0;

  instance->xprevious = malloc(rdA*sizeof(float));
  if ( instance->xprevious==NULL )
    printf("Unable to allocate memory\n");

  instance->searchPoint = malloc(rdA*sizeof(float));
  if ( instance->searchPoint==NULL )
    printf("Unable to allocate memory\n");
  cblas_scopy(rdA, instance->xcurrent, 1, instance->searchPoint, 1);

  instance->gradvalue = malloc(rdA*sizeof(float));
  instance->eta = malloc((ldA+rdA)*sizeof(float));
  if ( instance->gradvalue==NULL || instance->eta==NULL )
    printf("Unable to allocate memory\n");

  //Initialize stop values:
  int iter=0;
  float xdiff=1;
  float funcdiff=1;

  while(iter < MAX_ITER && xdiff > MIN_XDIFF && funcdiff > MIN_FUNCDIFF)
    {
      cblas_scopy(rdA, instance->xcurrent, 1, instance->xprevious, 1); //set xprevious to xcurrent

      //RUN BACKTRACKING ROUTINE
      ISTAbacktrack( instance );

      //UPDATE TERMINATING VALUES
      cblas_saxpy(rdA, -1.0, instance->xcurrent, 1, instance->xprevious, 1); //xprevious now holds "xprevious - xcurrent"
      xdiff = cblas_snrm2(rdA, instance->xprevious, 1);

      funcdiff = ISTAregress_func(instance->searchPoint, instance) - ISTAregress_func(instance->xcurrent, instance);
      funcdiff += instance->lambda * cblas_sasum(rdA, instance->searchPoint, 1);
      funcdiff -= instance->lambda * cblas_sasum(rdA, instance->xcurrent, 1);

      //UPDATE SEARCHPOINT
      if( instance->acceleration ) //FISTA searchpoint
	{
	  cblas_sscal(rdA, - iter / (float)(iter + 2), instance->xprevious, 1);
	  cblas_saxpy(rdA, 1.0, instance->xcurrent, 1, instance->xprevious, 1); //now xprevious equals what we want
	  cblas_scopy(rdA, instance->xprevious, 1, instance->searchPoint, 1);
	}
      else //regular ISTA searchpoint
	{
	  cblas_scopy(rdA, instance->xcurrent, 1, instance->searchPoint, 1);
	}

      //UPDATE ITERATOR
      iter++;
    }
  
  printf("iter: %d xdiff: %f funcdiff: %f\n", iter, xdiff, funcdiff);
  printf("final regression function value: %f\n", ISTAregress_func(instance->xcurrent, instance) );

  //FREE MEMORY
  free(instance->eta); free(instance->gradvalue); free(instance->searchPoint); free(instance->xprevious);
  free(instance->stepsize); free(instance);
									       
}

void ISTAsolve_lite(ISTAinstance* instance, int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF )
{
  // This version of ISTAsolve solve does not allocate any memory

  //Initialize stop values:
  int iter=0;
  float xdiff=1;
  float funcdiff=1;

  while(iter < MAX_ITER && xdiff > MIN_XDIFF && funcdiff > MIN_FUNCDIFF)
    {
      cblas_scopy(instance->rdA, instance->xcurrent, 1, instance->xprevious, 1); //set xprevious to xcurrent

      //RUN BACKTRACKING ROUTINE
      ISTAbacktrack( instance );

      //UPDATE TERMINATING VALUES
      cblas_saxpy(instance->rdA, -1.0, instance->xcurrent, 1, instance->xprevious, 1); //xprevious now holds "xprevious - xcurrent"
      xdiff = cblas_snrm2(instance->rdA, instance->xprevious, 1);

      funcdiff = ISTAregress_func(instance->searchPoint, instance) - ISTAregress_func(instance->xcurrent, instance);
      funcdiff += instance->lambda * cblas_sasum(instance->rdA, instance->searchPoint, 1);
      funcdiff -= instance->lambda * cblas_sasum(instance->rdA, instance->xcurrent, 1);

      //UPDATE SEARCHPOINT
      if( instance->acceleration ) //FISTA searchpoint
	{
	  cblas_sscal(instance->rdA, - iter / (float)(iter + 2), instance->xprevious, 1);
	  cblas_saxpy(instance->rdA, 1.0, instance->xcurrent, 1, instance->xprevious, 1); //now xprevious equals what we want
	  cblas_scopy(instance->rdA, instance->xprevious, 1, instance->searchPoint, 1);
	}
      else //regular ISTA searchpoint
	{
	  cblas_scopy(instance->rdA, instance->xcurrent, 1, instance->searchPoint, 1);
	}

      //UPDATE ITERATOR
      iter++;
    }

}

float** ISTAsolve_pathwise(float* lambdas, int num_lambdas, ISTAinstance* instance, 
			   int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF )
{
  // lambdas is an array of floats, ordered from largest to smallest, giving the lambdas we will solve for.
  // for each lambda in lambdas, we run ISTA and return the final xvalue

  float** values = malloc( num_lambdas*sizeof(float*) );
  if( values==NULL )
    printf("Unable to allocate memory");

  int i;

  for( i=0; i < num_lambdas; i++)
    {
      instance->lambda = lambdas[i];

      // Solve with new lambda value
      ISTAsolve_lite( instance, MAX_ITER, MIN_XDIFF, MIN_FUNCDIFF );

      // record solution in values[i]
      values[i] = malloc( (instance->rdA) * sizeof(float) );
      if ( values[i] == NULL )
	printf("Unable to allocate memory");
      cblas_scopy(instance->rdA, instance->xcurrent, 1, values[i], 1); 
    }

  return values;
}

void ISTAbacktrack(ISTAinstance* instance)
{
  /* initialize */
  int i;
  int numTrials = 0;
  float difference;

  /* calculate gradient at current searchPoint */
  ISTAgrad(instance);
  
  do
  {
    if(numTrials > 0) /*dont update stepsize the first time through */
      *(instance->stepsize) *= instance->gamma;

    /*update xcurrent = soft(  searchPoint - stepsize*gradvalue , lambda*stepsize )  */
    cblas_scopy(instance->rdA, instance->searchPoint, 1, instance->xcurrent, 1);
    cblas_saxpy(instance->rdA, -(*(instance->stepsize)), instance->gradvalue, 1, instance->xcurrent, 1);
    soft_threshold(instance->xcurrent, instance->rdA, instance->lambda * (*(instance->stepsize)));

    /*calculate difference that, when negative, guarantees the objective function decreases */
    difference = ISTAregress_func(instance->xcurrent, instance) - 
	         ISTAregress_func(instance->searchPoint, instance);
    cblas_scopy(instance->rdA, instance->xcurrent, 1, instance->eta, 1);
    cblas_saxpy(instance->rdA, -1.0, instance->searchPoint, 1, instance->eta, 1); //eta now holds "xcurrent - searchpoint"
    difference -= cblas_sdot(instance->rdA, instance->eta, 1, instance->gradvalue, 1);
    difference -= cblas_sdot(instance->rdA, instance->eta, 1, instance->eta, 1) / (2 * (*(instance->stepsize)) );

    numTrials++;

  } while(numTrials < 100 && difference > 0);
  
  if(numTrials == 100)
    printf("backtracking failed\n");

}


void ISTAgrad(ISTAinstance* instance)
{
  // THIS FUNCTION CALCULATES THE GRADIENT OF THE SMOOTH FUNCTION ISTAregress_func
  // AT THE POINT "searchPoint" and stores it in "gradvalue"

  int i;
  
  switch ( instance->regressionType )
    {
    case 'l': /*Here we calculate the gradient: 2*A'*(A*searchPoint - b)  */
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  instance->searchPoint, 1, 0.0, instance->eta, 1);
      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*searchPoint - b

      cblas_sgemv(CblasRowMajor, CblasTrans, instance->ldA, instance->rdA, 2.0, instance->A, instance->rdA,
		  instance->eta, 1, 0.0, instance->gradvalue, 1);
      break;

    case 'o': /*Here we calculate the gradient:A'*(p(A*searchPoint) - b) where p is the logistic function */
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  instance->searchPoint, 1, 0.0, instance->eta, 1);
      for(i=0; i < instance->ldA; i++)
	{
	  instance->eta[i] = 1 / (1 + exp( -(instance->eta[i]) ) ) - instance->b[i];
	}

      cblas_sgemv(CblasRowMajor, CblasTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA,
		  instance->eta, 1, 0.0, instance->gradvalue, 1);
      break;

    }

}

float ISTAregress_func(float* xvalue, ISTAinstance* instance)
{
  //THIS FUNCTION REPRESENTS THE SMOOTH FUNCTION THAT WE ARE TRYING TO OPTIMIZE
  //WHILE KEEPING THE REGULARIZATION FUNCTION (USUALLY THE 1-NORM) SMALL

  int i;
  float value = 0;

  switch (instance->regressionType) 
    {
    case 'l': /*In this case, the regression function is ||A*xvalue - b||^2 */
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  xvalue, 1, 0.0, instance->eta, 1);
      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*xvalue - b

      value = pow( cblas_snrm2(instance->ldA, instance->eta, 1 ), 2);
      break;

    case 'o': /*Regression function: sum log(1+ e^(A_i * x)) - A_i * x * b_i */
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  xvalue, 1, 0.0, instance->eta, 1);
      for(i=0; i < instance->ldA; i++)
	{
	  value += log( 1 + exp( instance->eta[i] )) - instance->eta[i] * instance->b[i];
	}
      break;
    }
      
  return value;
}

void soft_threshold(float* xvalue, int xlength, float threshold)
{
  //IMPLEMENTATION OF THE SOFT THRESHOLDING OPERATION

  if( threshold < 0)
    {
       printf("threshold value should be nonnegative\n");
       return;
    }
  int i;
  for( i=0; i < xlength; i++)
    {
      if(xvalue[i] > threshold)
	xvalue[i] -= threshold;
      else if(xvalue[i] < -threshold)
	xvalue[i] += threshold;
      else
	xvalue[i] = 0;
    }
}
