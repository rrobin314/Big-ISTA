#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cblas.h>
#include"istalib.h"


void ISTAbacktrack(float* A, int ldA, int rdA, float*b, float* searchPoint, float* xcurrent, 
		float* stepsize, float lambda, float gamma, char regressionType)
{
  /* initialize */
  int i;
  int numTrials = 0;
  float *gradvalue;
  float difference;

  /* calculate gradient */
  gradvalue = ISTAgrad(searchPoint, A, ldA, rdA, b, regressionType);
  
  do
  {
    if(numTrials > 0) /*dont update stepsize the first time through */
      (*stepsize) *= gamma;

    /*update xcurrent */
    for(i=0; i<rdA; i++)
      xcurrent[i] = searchPoint[i] - (*stepsize)*gradvalue[i]; 
    soft_threshold(xcurrent, rdA, lambda*(*stepsize)); //Thresholding that differentiates this as ISTA

    /*calculate difference that, when negative, guarantees the objective function decreases */
    difference = ISTAregress_func(xcurrent, A, ldA, rdA, b, regressionType) - 
	         ISTAregress_func(searchPoint, A, ldA, rdA, b, regressionType);
    for(i=0; i<rdA; i++)
      {
      difference -= (xcurrent[i] - searchPoint[i]) * gradvalue[i];
      difference -= (xcurrent[i] - searchPoint[i])*(xcurrent[i] - searchPoint[i]) / (2 * (*stepsize) );
      }

     numTrials++;
  } while(numTrials < 100 && difference > 0);
  
  if(numTrials == 100)
    printf("backtracking failed\n");

  free(gradvalue);
}


float* ISTAgrad(float* xvalue, float* A, int ldA, int rdA, float* b, char regressionType)
{
  // THIS FUNCTION CALCULATES THE GRADIENT OF THE SMOOTH FUNCTION ISTAregress_func

  float *gradvalue, *intermediatevalue;
  gradvalue = (float*)calloc(rdA, sizeof(float));
  intermediatevalue = (float*)calloc(ldA, sizeof(float));
  if ( gradvalue==NULL || intermediatevalue==NULL )
    printf("Unable to allocate memory\n");
  
  switch (regressionType)
    {
    case 'l': /*Here we calculate the gradient: 2*A'*(A*xvalue - b)  */
      cblas_sgemv(CblasRowMajor, CblasNoTrans, ldA, rdA, 1.0, A, rdA, 
		  xvalue, 1, 0.0, intermediatevalue, 1);
      int i;
      for(i=0; i<ldA; i++)
	{
	  intermediatevalue[i] -= b[i];
	}
      cblas_sgemv(CblasRowMajor, CblasTrans, ldA, rdA, 2.0, A, rdA,
		  intermediatevalue, 1, 0.0, gradvalue, 1);
      break;

    case 'o': /*Here we calculate the gradient:A'*(p(A*xvalue) - b) where p is the logistic function */
      for(i=0; i<ldA; i++)
	{
	  intermediatevalue[i] = exp( - cblas_sdot(rdA, &A[i*rdA], 1, xvalue, 1) ); 
	  intermediatevalue[i] = 1 / (1+intermediatevalue[i]) - b[i];
	}
      cblas_sgemv(CblasRowMajor, CblasTrans, ldA, rdA, 1.0, A, rdA,
		  intermediatevalue, 1, 0.0, gradvalue, 1);
      break;

    }
  
  free(intermediatevalue);
  return gradvalue;
}

float ISTAregress_func(float* xvalue, float* A, int ldA, int rdA, float* b, char regressionType)
{
  //THIS FUNCTION REPRESENTS THE SMOOTH FUNCTION THAT WE ARE TRYING TO OPTIMIZE
  //WHILE KEEPING THE REGULARIZATION FUNCTION (USUALLY THE 1-NORM) SMALL

  int i;
  float value = 0;
  switch (regressionType) 
    {
    case 'l': /*In this case, the regression function is ||A*xvalue - b||^2 */
      for(i=0; i<ldA; i++)
	{
	  value += pow( cblas_sdot(rdA, &A[i*rdA], 1, xvalue, 1) - b[i], 2 );
	}
      break;

    case 'o': /*Regression function: sum log(1+ e^(A_i * x)) - A_i * x * b_i */
      for(i=0; i<ldA; i++)
	{
	  value += log( 1 + exp( cblas_sdot(rdA, &A[i*rdA], 1, xvalue, 1) )) - cblas_sdot(rdA, &A[i*rdA], 1, xvalue, 1)*b[i];
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
