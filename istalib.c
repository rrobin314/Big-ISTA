#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cblas.h>
#include"istalib.h"

ISTAinstance* ISTAinstance_new(float* A, int ldA, int rdA, float* b, float lambda, float gamma, 
			       int acceleration, int interceptFlag, char regressionType, float* xvalue, float step )
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
  instance->interceptFlag = interceptFlag;
  instance->regressionType = regressionType;
  instance->intercept = 0.0;
  instance->xcurrent = xvalue;

  //ALLOCATE MEMORY
  instance->meanShifts = malloc(rdA*sizeof(float));
  instance->scalingFactors = malloc(rdA*sizeof(float));
  if ( instance->meanShifts==NULL || instance->scalingFactors==NULL )
    printf("Unable to allocate memory\n");

  instance->stepsize = malloc(sizeof(float));
  *(instance->stepsize) = step;

  instance->xprevious = malloc((rdA+1)*sizeof(float));
  if ( instance->xprevious==NULL )
    printf("Unable to allocate memory\n");

  instance->searchPoint = malloc((rdA+1)*sizeof(float));
  if ( instance->searchPoint==NULL )
    printf("Unable to allocate memory\n");
  cblas_scopy(rdA, instance->xcurrent, 1, instance->searchPoint, 1);

  instance->gradvalue = malloc((rdA+1)*sizeof(float));
  instance->eta = malloc((ldA+rdA)*sizeof(float));
  if ( instance->gradvalue==NULL || instance->eta==NULL )
    printf("Unable to allocate memory\n");

  //OUTPUT MESSAGE
  fprintf(stdout,"Created ISTA instance with parameters:\n ldA: %d rdA: %d lambda: %f gamma: %f accel: %d regType: %c step: %f \n A[0]: %f A[last]: %f \n b[0]: %f b[last]: %f \n x[0]: %f x[last]: %f \n", ldA, rdA, lambda, gamma, acceleration,
	  regressionType, step, A[0], A[(rdA+1)*ldA-2], b[0], b[ldA-1], xvalue[0], xvalue[rdA-1]);

  if(interceptFlag) {
    fprintf(stdout, "Added column of ones to A for intercept\n");
  }
  else {
    fprintf(stdout, "A is being used unchanged (no column of ones added)\n");
  }

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
  free(instance->scalingFactors);
  free(instance->meanShifts);
  free(instance-> b);
  free(instance-> A);
  free(instance);
}

extern void ISTArescale(ISTAinstance* instance)
{
  int i, j;
  float mean, norm;

  // LOOP THROUGH THE COLUMNS OF A (NOT INCLUDING THE (STILL ZERO) INTERCEPT COLUMN)
  for(j=0; j<(instance->rdA); j++) {

    //CALCULATE MEAN OF THE COLUMN 
    mean = 0;
    for(i=0; i<(instance->ldA); i++) {
      mean += instance->A[i*(instance->rdA + 1) + j];
    }
    mean = mean / (instance->ldA);

    //SUBTRACT MEAN FROM THE COLUMN
    for(i=0; i<(instance->ldA); i++) {
      instance->A[i*(instance->rdA + 1) + j] -= mean;
    }

    //STORE MEAN
    instance->meanShifts[j] = mean;

    //CALCULATE 2-NORM OF COLUMN
    norm = cblas_snrm2(instance->ldA, instance->A + j, instance->rdA + 1);

    //SCALE COLUMN TO HAVE UNIT NORM
    if(norm > 0.0001)
      cblas_sscal(instance->ldA, 1.0 / norm, instance->A + j, instance->rdA + 1);

    //STORE NORM
    instance->scalingFactors[j] = norm;

  }
}

extern void ISTAundoRescale(ISTAinstance* instance)
{
  int i;

  //TRANSFER SCALING FACTORS ONTO THE X VALUES
  for(i=0; i<(instance->rdA); i++) {
    if(instance->scalingFactors[i] > 0.0001)
      instance->xcurrent[i] = instance->xcurrent[i] / instance->scalingFactors[i];
  }

  //CALCULATE INTERCEPT SO THAT "intercept" + "unscaled A" * xcurrent = b
  instance->intercept = -1.0 * cblas_sdot(instance->rdA, instance->xcurrent, 1, instance->meanShifts, 1);

}

extern void ISTAaddIntercept(ISTAinstance* instance) 
{
  int i;
  if(instance->interceptFlag) {
    for(i=0; i<(instance->ldA); i++) {
      //CHANGE FINAL COLUMN OF A TO ONES
      instance->A[(i+1)*(instance->rdA + 1) - 1] = 1;
    }
  }
}


void ISTAsolve_lite(ISTAinstance* instance, int MAX_ITER, float MIN_FUNCDIFF )
{
  // This version of ISTAsolve solve does not allocate any memory

  //Initialize stop values:
  int iter=0;
  int i;
  float funcdiff=1;

  printf("intial objective function value for lambda %f: %f\n", instance->lambda, ISTAregress_func(instance->xcurrent, instance) + instance->lambda * cblas_sasum(instance->rdA, instance->xcurrent, 1) );

  while(iter < MAX_ITER && funcdiff > MIN_FUNCDIFF)
    {
      cblas_scopy(instance->rdA + 1, instance->xcurrent, 1, instance->xprevious, 1); //set xprevious to xcurrent

      //RUN BACKTRACKING ROUTINE
      ISTAbacktrack( instance );

      //UPDATE TERMINATING VALUES
      cblas_saxpy(instance->rdA + 1, -1.0, instance->xcurrent, 1, instance->xprevious, 1); //xprevious now holds "xprevious - xcurrent"
      //xdiff = cblas_snrm2(instance->rdA, instance->xprevious, 1);

      /*funcdiff = ISTAregress_func(instance->searchPoint, instance) - ISTAregress_func(instance->xcurrent, instance);
      funcdiff += instance->lambda * cblas_sasum(instance->rdA, instance->searchPoint, 1);
      funcdiff -= instance->lambda * cblas_sasum(instance->rdA, instance->xcurrent, 1);
      */
      funcdiff = ISTAregress_func(instance->xcurrent, instance) + instance->lambda * cblas_sasum(instance->rdA, instance->xcurrent, 1);
      funcdiff = funcdiff / (ISTAregress_func(instance->searchPoint, instance) + instance->lambda * cblas_sasum(instance->rdA, instance->searchPoint, 1) );
      funcdiff = 1 - funcdiff;

      //UPDATE SEARCHPOINT
      if( instance->acceleration ) //FISTA searchpoint
	{
	  cblas_sscal(instance->rdA + 1, - iter / (float)(iter + 2), instance->xprevious, 1);
	  cblas_saxpy(instance->rdA + 1, 1.0, instance->xcurrent, 1, instance->xprevious, 1); //now xprevious equals what we want
	  cblas_scopy(instance->rdA + 1, instance->xprevious, 1, instance->searchPoint, 1);
	}
      else //regular ISTA searchpoint
	{
	  cblas_scopy(instance->rdA + 1, instance->xcurrent, 1, instance->searchPoint, 1);
	}

      //DEBUGGING
      /*if(iter <= 1 && instance->lambda >= 16) {
	fprintf(stdout, "\ngradient: ");
	for(i=0; i<5; i++) {
	  fprintf(stdout, "%f ", instance->gradvalue[i]);
	}
	fprintf(stdout, "\nsearchpoint: ");
	for(i=0; i<5; i++) {
	  fprintf(stdout, "%f ", instance->searchPoint[i]);
	}
	fprintf(stdout, "\n");
	}*/

      //UPDATE ITERATOR
      iter++;
    }
  printf("iter: %d funcdiff: %f\n", iter, funcdiff);
  printf("final objective function value for lambda %f: %f\n", instance->lambda, ISTAregress_func(instance->xcurrent, instance) + instance->lambda * cblas_sasum(instance->rdA, instance->xcurrent, 1) );

}

void ISTAbacktrack(ISTAinstance* instance)
{
  /* initialize */
  int i;
  int numTrials = 0;
  float difference;

  // calculate gradient at current searchPoint 
  ISTAgrad(instance);
  
  do
  {
    if(numTrials > 0) /*dont update stepsize the first time through */
      *(instance->stepsize) *= instance->gamma;

    //update xcurrent = soft(  searchPoint - stepsize*gradvalue , lambda*stepsize )  
    cblas_scopy(instance->rdA + 1, instance->searchPoint, 1, instance->xcurrent, 1);
    cblas_saxpy(instance->rdA + 1, -(*(instance->stepsize)), instance->gradvalue, 1, instance->xcurrent, 1);
    //ONLY THRESHOLD THE FIRST rdA ENTRIES OF xcurrent; DONT TOUCH THE INTERCEPT ENTRY
    soft_threshold(instance->xcurrent, instance->rdA, instance->lambda * (*(instance->stepsize)));

    //calculate difference that, when negative, guarantees the objective function decreases 
    difference = ISTAregress_func(instance->xcurrent, instance) - 
	         ISTAregress_func(instance->searchPoint, instance);
    cblas_scopy(instance->rdA + 1, instance->xcurrent, 1, instance->eta, 1);
    cblas_saxpy(instance->rdA + 1, -1.0, instance->searchPoint, 1, instance->eta, 1); //eta now holds "xcurrent - searchpoint"

    //DEBUGGING
    //fprintf(stdout, "eta[0] %f predifference %f \n", instance->eta[0], difference);

    difference -= cblas_sdot(instance->rdA + 1, instance->eta, 1, instance->gradvalue, 1);
    difference -= cblas_sdot(instance->rdA + 1, instance->eta, 1, instance->eta, 1) / (2 * (*(instance->stepsize)) );

    numTrials++;

    //DEBUGGING
    //fprintf(stdout, "xcurrent[0] %f difference %f \n", instance->xcurrent[0], difference);

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
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA + 1, 1.0, instance->A, instance->rdA + 1, 
		  instance->searchPoint, 1, 0.0, instance->eta, 1);
      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*searchPoint - b

      cblas_sgemv(CblasRowMajor, CblasTrans, instance->ldA, instance->rdA + 1, 2.0, instance->A, instance->rdA + 1,
		  instance->eta, 1, 0.0, instance->gradvalue, 1);
      break;

    case 'o': /*Here we calculate the gradient:A'*(p(A*searchPoint) - b) where p is the logistic function */
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA + 1, 1.0, instance->A, instance->rdA + 1, 
		  instance->searchPoint, 1, 0.0, instance->eta, 1);
      for(i=0; i < instance->ldA; i++)
	{
	  instance->eta[i] = 1 / (1 + exp( -(instance->eta[i]) ) ) - instance->b[i];
	}

      cblas_sgemv(CblasRowMajor, CblasTrans, instance->ldA, instance->rdA + 1, 1.0, instance->A, instance->rdA + 1,
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
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA + 1, 1.0, instance->A, instance->rdA + 1, 
		  xvalue, 1, 0.0, instance->eta, 1);
      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*xvalue - b

      value = pow( cblas_snrm2(instance->ldA, instance->eta, 1 ), 2);
      break;

    case 'o': /*Regression function: sum log(1+ e^(A_i * x)) - A_i * x * b_i */
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA + 1, 1.0, instance->A, instance->rdA + 1, 
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

extern void calcLambdas(float* lambdas, int numLambdas, float lambdaStart, 
			float lambdaFinish, float* A, int ldA, int rdA, 
			float* b, float* result)
{
  float startValue;
  int j;

  //IF lambdaStart is negative, then do automatic calculation of startValue
  if(lambdaStart <= 0) {
    //result = A' * b
    cblas_sgemv(CblasRowMajor, CblasTrans, ldA, rdA+1, 1.0, A, rdA+1,
		b, 1, 0.0, result, 1);
    //i = index of max in absolute value of result
    //With lambda = result[i], 0 will be an optimal solution of our optimization
    CBLAS_INDEX i = cblas_isamax(rdA+1, result, 1);
    startValue = fabs(result[i]) / 2.0;
  }
  else if(lambdaStart > 0) {
    startValue = lambdaStart;
  }

  //Fill in lambdas with exponential path from startValue to lambdaFinish
  if(numLambdas == 1) {
    lambdas[0] = lambdaFinish;
  }
  else if(numLambdas >= 2) {
    for(j=0; j < numLambdas; j++) {
      lambdas[j] = startValue * exp( log(lambdaFinish / startValue) * j / (numLambdas - 1) );
      //lambdas[j] = startValue - j * (startValue - lambdaFinish) / (numLambdas - 1);
    }
  }

}
