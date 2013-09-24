#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include"istalib.h"

int main()
{
  srand(time(NULL));
  int i,j;
  int ldA = 10; //left dimension of A
  int rdA = 50; //right dimension of A
  float lambda = 0.05; //how much weight we give to the 1-norm
  float gamma = 0.9; //fraction we decrease stepsize each time it fails
  int accel = 0; //0 if normal ISTA update; 1 for FISTA acceleration
  char regType = 'l'; // 'l' for linear regression and 'o' for logistic regression
  int MAX_ITER = 10000;
  int MIN_XDIFF = 0.00001;
  int MIN_FUNCDIFF = 0.00001;

  //INITIALIZE A and b to random entries
  float* A = calloc(ldA*rdA, sizeof(float));
  float* b = calloc(ldA, sizeof(float));
  float* xvalue = calloc(rdA, sizeof(float));
  for( i=0; i<ldA*rdA; i++)
    A[i] = (float)rand()/(float)RAND_MAX;
  for( i=0; i<ldA; i++)
    b[i] = (float)rand()/(float)RAND_MAX;

  //  for( i=0; i<rdA; i++)
  //  printf("%f\n", xvalue[i]);
  /*
  ISTAsolve(A, ldA, rdA, b, lambda, gamma, 
	    accel, regType, xvalue, 
	    MAX_ITER, MIN_XDIFF, MIN_FUNCDIFF);

  for( i=0; i<rdA; i++)
    printf("%f\n", xvalue[i]);

  free(A); free(b); free(xvalue);
  */

  ISTAinstance* INST = ISTAinstance_new(A, ldA, rdA, b, lambda, gamma, accel, regType, xvalue, 1.0);
  float** values;
  float lambdas[] = {5, 4.7, 4.4, 3.7};

  values = ISTAsolve_pathwise(lambdas, 4, INST, MAX_ITER, MIN_XDIFF, MIN_FUNCDIFF );

  for(i=0; i<4; i++)
    {
      printf("\n value %d: \n", i);

      for(j=0; j<rdA; j++)
	{
	  printf("%f\n", values[i][j]);
	}
    }

  for(i=0; i<4; i++)
    free(values[i]);
  free(values);
  ISTAinstance_free(INST);      

  return 0;
}
