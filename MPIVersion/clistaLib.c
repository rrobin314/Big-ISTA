#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cblas.h>
#include"mpi.h"
#include"clistaLib.h"

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

      funcdiff = ISTAloss_func_mpi(instance->searchPoint, instance) - ISTAloss_func_mpi(instance->xcurrent, instance);
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
  printf("final regression function value: %f\n", ISTAloss_func_mpi(instance->xcurrent, instance) );

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

      funcdiff = ISTAloss_func_mpi(instance->searchPoint, instance) - ISTAloss_func_mpi(instance->xcurrent, instance);
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

float ISTAcrossval(ISTAinstance* instance, int* folds, int num_folds, 
		   int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF )
{
  float meanTotalError = 0.0;
  int i, j, iter, currentFold;
  float xdiff, funcdiff, foldError;

  // for each fold number, we run ISTA on the rows not included in that fold
  for( currentFold=0; currentFold < num_folds; currentFold++)
    {
      
      iter=0;
      xdiff = 1.0;
      funcdiff = 1.0;
      foldError = 0.0;

      int numRowsInFold = 0;
      for(i=0; i < instance->ldA; i++)
	{
	  if(currentFold == folds[i])
	    numRowsInFold++;
	}
      if(numRowsInFold == 0)
	printf("no rows in this folds!");

      // reset xcurrent to 0 vector
      cblas_sscal(instance->rdA, 0.0, instance->xcurrent, 1);

      // Now run the usual ISTA solve code, but calling the crossval specific functions
      while(iter < MAX_ITER && xdiff > MIN_XDIFF && funcdiff > MIN_FUNCDIFF)
	{
	  cblas_scopy(instance->rdA, instance->xcurrent, 1, instance->xprevious, 1); //set xprevious to xcurrent
	  
	  //RUN BACKTRACKING ROUTINE
	  ISTAbacktrack_cv( instance, currentFold, folds);
	  
	  //UPDATE TERMINATING VALUES
	  cblas_saxpy(instance->rdA, -1.0, instance->xcurrent, 1, instance->xprevious, 1); //xprevious now holds "xprevious - xcurrent"
	  xdiff = cblas_snrm2(instance->rdA, instance->xprevious, 1);
	  
	  funcdiff = ISTAregress_func_cv(instance->searchPoint, instance, currentFold, folds, 0)
	              - ISTAregress_func_cv(instance->xcurrent, instance, currentFold, folds, 0);
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
       

      // Now calculate the average test error on the rows in the fold
      if(numRowsInFold != 0)
	foldError = ISTAregress_func_cv(instance->xcurrent, instance, currentFold, folds, 1) / numRowsInFold;
      
      // Update meanTotalError
      meanTotalError += foldError;

    }

  meanTotalError = meanTotalError / num_folds;
  return meanTotalError;
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
    difference = ISTAloss_func_mpi(instance->xcurrent, instance) - 
	         ISTAloss_func_mpi(instance->searchPoint, instance);
    cblas_scopy(instance->rdA, instance->xcurrent, 1, instance->eta, 1);
    cblas_saxpy(instance->rdA, -1.0, instance->searchPoint, 1, instance->eta, 1); //eta now holds "xcurrent - searchpoint"
    difference -= cblas_sdot(instance->rdA, instance->eta, 1, instance->gradvalue, 1);
    difference -= cblas_sdot(instance->rdA, instance->eta, 1, instance->eta, 1) / (2 * (*(instance->stepsize)) );

    numTrials++;

  } while(numTrials < 100 && difference > 0);
  
  if(numTrials == 100)
    printf("backtracking failed\n");

}

void ISTAbacktrack_cv(ISTAinstance* instance, int currentFold, int* folds)
{
  /* initialize */
  int i;
  int numTrials = 0;
  float difference;

  /* calculate gradient at current searchPoint */
  ISTAgrad_cv(instance, currentFold, folds);
  
  do
  {
    if(numTrials > 0) /*dont update stepsize the first time through */
      *(instance->stepsize) *= instance->gamma;

    /*update xcurrent = soft(  searchPoint - stepsize*gradvalue , lambda*stepsize )  */
    cblas_scopy(instance->rdA, instance->searchPoint, 1, instance->xcurrent, 1);
    cblas_saxpy(instance->rdA, -(*(instance->stepsize)), instance->gradvalue, 1, instance->xcurrent, 1);
    soft_threshold(instance->xcurrent, instance->rdA, instance->lambda * (*(instance->stepsize)));

    /*calculate difference that, when negative, guarantees the objective function decreases */
    difference = ISTAregress_func_cv(instance->xcurrent, instance, currentFold, folds, 0) - 
	         ISTAregress_func_cv(instance->searchPoint, instance, currentFold, folds, 0);
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

void ISTAgrad_cv(ISTAinstance* instance, int currentFold, int* folds)
{
  // THIS FUNCTION CALCULATES THE GRADIENT OF THE SMOOTH FUNCTION ISTAregress_func
  // AT THE POINT "searchPoint" and stores it in "gradvalue" ONLY USING THE ROWS
  // OF A AND b CORRESPONDING TO THOSE NOT IN THE CURRENT FOLD

  int i;
  
  switch ( instance->regressionType )
    {
    case 'l': /*Here we calculate the gradient: 2*A'*(A*searchPoint - b)  */
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  instance->searchPoint, 1, 0.0, instance->eta, 1);
      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*searchPoint - b

      // kill those entries of eta that lie in the fold (test set)
      for(i=0; i < instance->ldA; i++)
	{
	  if(folds[i] == currentFold)
	    instance->eta[i] = 0.0;
	}

      cblas_sgemv(CblasRowMajor, CblasTrans, instance->ldA, instance->rdA, 2.0, instance->A, instance->rdA,
		  instance->eta, 1, 0.0, instance->gradvalue, 1);
      break;

    case 'o': /*Here we calculate the gradient:A'*(p(A*searchPoint) - b) where p is the logistic function */
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  instance->searchPoint, 1, 0.0, instance->eta, 1);
      for(i=0; i < instance->ldA; i++)
	{
	  if(folds[i] == currentFold)
	    instance->eta[i] = 0.0;
	  else
	    instance->eta[i] = 1 / (1 + exp( -(instance->eta[i]) ) ) - instance->b[i];
	}

      cblas_sgemv(CblasRowMajor, CblasTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA,
		  instance->eta, 1, 0.0, instance->gradvalue, 1);
      break;

    }

}

float ISTAloss_func_mpi(float* xvalue, ISTAinstance* instance)
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

float ISTAregress_func_cv(float* xvalue, ISTAinstance* instance, int currentFold, int* folds, int insideFold)
{
  // If insideFold==1, then we only consider those rows inside the fold
  // If insideFold==0, then we consider those rows outside the fold

  int j;
  float value = 0;

  switch (instance->regressionType) 
    {
    case 'l': /*In this case, the regression function is ||A*xvalue - b||^2 */
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  xvalue, 1, 0.0, instance->eta, 1);
      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*xvalue - b

      // Now kill all elements of eta that we don't want
      for(j=0; j < instance->ldA; j++)
	{
	  if( (folds[j] != currentFold && insideFold == 1) || (folds[j] == currentFold && insideFold == 0) )
	    instance->eta[j] = 0.0;
	}

      value = pow( cblas_snrm2(instance->ldA, instance->eta, 1 ), 2);
      break;

    case 'o': /*Regression function: sum log(1+ e^(A_i * x)) - A_i * x * b_i */
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  xvalue, 1, 0.0, instance->eta, 1);
      for(j=0; j < instance->ldA; j++)
	{
	  if( (folds[j] == currentFold && insideFold == 1) || (folds[j] != currentFold && insideFold == 0) )
	    value += log( 1 + exp( instance->eta[j] )) - instance->eta[j] * instance->b[j];
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

extern void multiply_Ax(float* xvalue, int lenx, int ldA, float* result, int nslaves, MPI_Comm comm, int TAG)
{
  int rank;
  float* temp = calloc(ldA, sizeof(float));
  if(temp==NULL)
    fprintf(stdout,"Unable to allocate memory!");

  for(rank=1; rank <= nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, TAG, comm);
    }

  MPI_Bcast(xvalue, lenx, MPI_FLOAT, 0, comm);

  MPI_Gather(temp, ldA, MPI_FLOAT, result, ldA, MPI_FLOAT, 0, comm);

  free(temp);

  return;
}

extern void multiply_ATAx(float* xvalue, int lenx, float* result, int nslaves, MPI_Comm comm, int TAG)
{
  int rank;
  float* temp = calloc(lenx, sizeof(float));
  if(temp==NULL)
    fprintf(stdout,"Unable to allocate memory!");


  //Tell slaves we are going to do A^t*A*xvalue
  for(rank=1; rank <= nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, TAG, comm);
    }

  MPI_Bcast(xvalue, lenx, MPI_FLOAT, 0, comm);

  MPI_Reduce(temp, result, lenx, MPI_FLOAT, MPI_SUM, 0, comm);

  free(temp);

  return;
}

extern void get_dat_matrix(float* A, int ldA, int rdA, int myrank)
{
  int i;

  fprintf(stdout,"Slave %d getting %d by %d matrix\n", myrank, ldA, rdA);

  for( i=0; i<ldA*rdA; i++)
    A[i] = myrank * (i+1) * 0.1;

  return;
}
