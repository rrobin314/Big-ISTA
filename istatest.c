#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include"istalib.h"

int main()
{
  srand(time(NULL));
  int i;
  int ldA = 10; //left dimension of A
  int rdA = 50; //right dimension of A
  float lambda = 0.1; //how much weight we give to the 1-norm
  float gamma = 0.9; //fraction we decrease stepsize each time it fails
  int accel = 0; //0 if normal ISTA update; 1 for FISTA acceleration
  char regType = 'l'; // 'l' for linear regression and 'o' for logistic regression

  //INITIALIZE A and b to random entries
  float* A = calloc(ldA*rdA, sizeof(float));
  float* b = calloc(ldA, sizeof(float));
  float* x0 = calloc(rdA, sizeof(float));
  float* xcurrent = calloc(rdA, sizeof(float));
  float* xprevious = calloc(rdA, sizeof(float));
  float* searchPoint = calloc(rdA, sizeof(float));
  for( i=0; i<ldA*rdA; i++)
    A[i] = (float)rand()/(float)RAND_MAX;
  for( i=0; i<ldA; i++)
    b[i] = (int)rand() % 2;

  int iter=0;
  float xdiff = 1;
  float funcdiff = 1;
  float* stepsize = malloc(sizeof(float));
  (*stepsize) = 0.3;


  while(iter < 10000 && xdiff > 0.00001 && funcdiff > 0.00001)
  {
    for( i=0; i<rdA; i++)
      xprevious[i] = xcurrent[i];

    // RUN BACKTRACKING ROUTINE
    ISTAbacktrack(A, ldA, rdA, b, searchPoint, xcurrent, 
	        stepsize, lambda, gamma, regType);

    // UPDATE TERMINATING VALUES
    xdiff = 0;
    for( i=0; i<rdA; i++)
      xdiff += fabs(xcurrent[i] - xprevious[i]);

    funcdiff = ISTAregress_func(searchPoint, A, ldA, rdA, b, regType) - ISTAregress_func(xcurrent, A, ldA, rdA, b, regType);
    for( i=0; i<rdA; i++)
      funcdiff += lambda*( fabs(searchPoint[i]) - fabs(xcurrent[i]) );

    // UPDATE SEARCHPOINT
    if( accel ) /*FISTA SEARCHPOINT */
      {
	for( i=0; i<rdA; i++)
	  searchPoint[i] = xcurrent[i] + (float)iter / (float)(iter + 2) * (xcurrent[i] - xprevious[i]);
      }
    else /*Regular ISTA searchpoint */
      {
	for( i=0; i<rdA; i++)
	  searchPoint[i] = xcurrent[i];
      }

    //UPDATE ITERATOR
    iter++;
    //   printf("%d\n",iter);    
  }

  printf("iter: %d xdiff: %f funcdiff: %f\n", iter, xdiff, funcdiff);
  printf("final regression function value: %f\nvalues of x:\n", ISTAregress_func(xcurrent, A, ldA, rdA, b, regType) );
  for( i=0; i<rdA; i++)
    printf("%f\n", xcurrent[i]);

  free(A); free(b); free(x0); free(xcurrent); free(xprevious); free(searchPoint);
  free(stepsize);

  return 0;
}
