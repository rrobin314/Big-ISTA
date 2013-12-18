#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cblas.h>
#include"mpi.h"
#include"clistaLib.h"

ISTAinstance_mpi* ISTAinstance_mpi_new(int slave_ldA, int rdA, float* b, float lambda, 
				       float gamma, int acceleration, char regressionType, 
				       float* xvalue, float step,
				       int nslaves, MPI_Comm comm, int ax, int atx, int atax, int die)
{
  // This method initializes an ISTAinstance object
  ISTAinstance_mpi* instance = malloc(sizeof(ISTAinstance_mpi));
  if ( instance==NULL )
    fprintf(stdout, "Unable to allocate memory\n");

  instance->slave_ldA = slave_ldA;
  instance->ldA = slave_ldA * nslaves;
  instance->rdA = rdA;
  instance->b = b;
  instance->lambda = lambda;
  instance->gamma = gamma;
  instance->acceleration = acceleration;
  instance->regressionType = regressionType;
  instance->xcurrent = xvalue;

  instance->nslaves = nslaves;
  instance->comm = comm;
  instance->tag_ax = ax;
  instance->tag_atx = atx;
  instance->tag_atax = atax;
  instance->tag_die = die;
 
  //Allocate memory for values used during the calculation
  instance->stepsize = malloc(sizeof(float));
  *(instance->stepsize) = step;

  instance->xprevious = malloc((rdA+1)*sizeof(float));
  if ( instance->xprevious==NULL )
    fprintf(stdout, "Unable to allocate memory\n");

  instance->searchPoint = malloc((rdA+1)*sizeof(float));
  if ( instance->searchPoint==NULL )
    fprintf(stdout, "Unable to allocate memory\n");
  cblas_scopy(rdA+1, instance->xcurrent, 1, instance->searchPoint, 1);

  instance->gradvalue = malloc(rdA+1*sizeof(float));
  instance->eta = malloc((instance->ldA + rdA)*sizeof(float));
  if ( instance->gradvalue==NULL || instance->eta==NULL )
    fprintf(stdout, "Unable to allocate memory\n");

  instance->meanShifts = calloc(rdA, sizeof(float));
  instance->scalingFactors = calloc(rdA, sizeof(float));
  if ( instance->meanShifts==NULL || instance->scalingFactors==NULL )
    fprintf(stdout, "Unable to allocate memory\n");

  fprintf(stdout,"Created ISTA instance with parameters:\n nslaves: %d slave_ldA: %d ldA: %d rdA: %d \n lambda: %f gamma: %f accel: %d regType: %c step: %f  \n b[0]: %f b[last]: %f \n x[0]: %f x[last]: %f \n", nslaves, slave_ldA, instance->ldA, rdA, lambda, gamma, acceleration,
	  regressionType, step, b[0], b[instance->ldA-1], xvalue[0], xvalue[rdA-1]);

  return instance;
}


void ISTAinstance_mpi_free(ISTAinstance_mpi* instance)
{
  // Frees an entire ISTAinstance pointer
  free(instance->meanShifts);
  free(instance->scalingFactors);
  free(instance->eta); 
  free(instance->gradvalue); 
  free(instance->searchPoint); 
  free(instance->xprevious);
  free(instance->xcurrent);
  free(instance->stepsize); 
  free(instance-> b);
  free(instance);
}


void ISTAsolve_lite(ISTAinstance_mpi* instance, int MAX_ITER, float MIN_FUNCDIFF )
{
  // This version of ISTAsolve solve does not allocate any memory

  //Initialize stop values:
  int iter=0;
  float funcdiff=1;

  fprintf(stdout, "intial objective function value for lambda %f: %f\n", instance->lambda, ISTAloss_func_mpi(instance->xcurrent, instance) + instance->lambda * cblas_sasum(instance->rdA, instance->xcurrent, 1) );

  while(iter < MAX_ITER && funcdiff > MIN_FUNCDIFF)
    {
      cblas_scopy(instance->rdA + 1, instance->xcurrent, 1, instance->xprevious, 1); //set xprevious to xcurrent

      //RUN BACKTRACKING ROUTINE
      ISTAbacktrack( instance );

      //UPDATE TERMINATING VALUES
      cblas_saxpy(instance->rdA + 1, -1.0, instance->xcurrent, 1, instance->xprevious, 1); //xprevious now holds "xprevious - xcurrent"
      //xdiff = cblas_snrm2(instance->rdA, instance->xprevious, 1);

      funcdiff = ISTAloss_func_mpi(instance->xcurrent, instance) + instance->lambda * cblas_sasum(instance->rdA, instance->xcurrent, 1);
      funcdiff = funcdiff / (ISTAloss_func_mpi(instance->searchPoint, instance) + instance->lambda * cblas_sasum(instance->rdA, instance->searchPoint, 1) );
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

      //UPDATE ITERATOR
      iter++;
    }

  fprintf(stdout, "iter: %d funcdiff: %f\n", iter, funcdiff);
  fprintf(stdout, "final objective function value for lambda %f: %f\n", instance->lambda, ISTAloss_func_mpi(instance->xcurrent, instance) + instance->lambda * cblas_sasum(instance->rdA, instance->xcurrent, 1) );

}

void ISTAbacktrack(ISTAinstance_mpi* instance)
{
  // initialize 
  int numTrials = 0;
  float difference;

  // calculate gradient at current searchPoint 
  ISTAgrad(instance);
  
  do
  {
    if(numTrials > 0) // dont update stepsize the first time through 
      *(instance->stepsize) *= instance->gamma;

    //update xcurrent = soft(  searchPoint - stepsize*gradvalue , lambda*stepsize )  
    cblas_scopy(instance->rdA + 1, instance->searchPoint, 1, instance->xcurrent, 1);
    cblas_saxpy(instance->rdA + 1, -(*(instance->stepsize)), instance->gradvalue, 1, instance->xcurrent, 1);
    //ONLY THRESHOLD THE FIRST rdA ENTRIES OF xcurrent; DONT TOUCH THE INTERCEPT ENTRY
    soft_threshold(instance->xcurrent, instance->rdA, instance->lambda * (*(instance->stepsize)));

    //calculate difference that, when negative, guarantees the objective function decreases
    difference = ISTAloss_func_mpi(instance->xcurrent, instance) - 
	         ISTAloss_func_mpi(instance->searchPoint, instance);
    cblas_scopy(instance->rdA + 1, instance->xcurrent, 1, instance->eta, 1);
    cblas_saxpy(instance->rdA + 1, -1.0, instance->searchPoint, 1, instance->eta, 1); //eta now holds "xcurrent - searchpoint"
    difference -= cblas_sdot(instance->rdA + 1, instance->eta, 1, instance->gradvalue, 1);
    difference -= cblas_sdot(instance->rdA + 1, instance->eta, 1, instance->eta, 1) / (2 * (*(instance->stepsize)) );

    numTrials++;

  } while(numTrials < 100 && difference > 0);
  
  if(numTrials == 100)
    fprintf(stdout, "backtracking failed\n");

}


extern void ISTAgrad(ISTAinstance_mpi* instance)
{
  // THIS FUNCTION CALCULATES THE GRADIENT OF THE SMOOTH FUNCTION ISTAloss_func_mpi
  // AT THE POINT "searchPoint" and stores it in "gradvalue"

  int i;
  
  switch ( instance->regressionType )
    {
    case 'l': // Here we calculate the gradient: 2*A'*(A*searchPoint - b)  
      multiply_Ax(instance->searchPoint, instance->eta, instance);
      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*searchPoint - b

      //NOTE: WE ARE MISSING THE SCALAR 2 IN THIS CALCULATION, DOES THIS MATTER?
      //ANSWER: Yes, it matters a lot! Added an addition sscal call to fix.
      multiply_ATx(instance->eta, instance->gradvalue, instance);
      cblas_sscal(instance->rdA + 1, 2.0, instance->gradvalue, 1);
      break;
      
    case 'o': // Here we calculate the gradient:A'*(p(A*searchPoint) - b) where p is the logistic function 
      multiply_Ax(instance->searchPoint, instance->eta, instance);
      for(i=0; i < instance->ldA; i++)
	{
	  instance->eta[i] = 1 / (1 + exp( -(instance->eta[i]) ) ) - instance->b[i];
	}

      multiply_ATx(instance->eta, instance->gradvalue, instance);

      break;
      
    }

}




float ISTAloss_func_mpi(float* xvalue, ISTAinstance_mpi* instance)
{
  //THIS FUNCTION REPRESENTS THE SMOOTH FUNCTION THAT WE ARE TRYING TO OPTIMIZE
  //WHILE KEEPING THE REGULARIZATION FUNCTION (USUALLY THE 1-NORM) SMALL

  int i;
  float value = 0;

  switch (instance->regressionType) 
    {
    case 'l': //In this case, the regression function is ||A*xvalue - b||^2 
      multiply_Ax(xvalue, instance->eta, instance);
      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*xvalue - b

      value = pow( cblas_snrm2(instance->ldA, instance->eta, 1 ), 2);
      break;

    case 'o': // Regression function: sum log(1+ e^(A_i * x)) - A_i * x * b_i 
      multiply_Ax(xvalue, instance->eta, instance);
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
       fprintf(stdout, "threshold value should be nonnegative\n");
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
			float lambdaFinish, ISTAinstance_mpi* instance)
{
  float startValue;
  int j;

  //IF lambdaStart is negative, then do automatic calculation of startValue
  if(lambdaStart <= 0) {
    //eta = A' * b
    multiply_ATx(instance->b, instance->eta, instance);
    //i = index of max in absolute value of eta
    //With lambda = eta[i], 0 will be an optimal solution of our optimization
    CBLAS_INDEX i = cblas_isamax(instance->rdA + 1, instance->eta, 1);
    startValue = fabs(instance->eta[i]) / 2.0;
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

//NOTE: length of x must be rdA+1, result is of length ldA
extern void multiply_Ax(float* xvalue, float* result, ISTAinstance_mpi* instance)
{
  int rank;
  float* dummyFloat;
  int* counts = calloc(instance->nslaves + 1, sizeof(int));
  int* displacements = calloc(instance->nslaves + 1, sizeof(int));
  if(counts==NULL || displacements==NULL)
    fprintf(stdout,"Unable to allocate memory!");
  for(rank=1; rank <= instance->nslaves; rank++)
    {
      counts[rank] = instance->slave_ldA;
      displacements[rank] = instance->slave_ldA*(rank-1);
    }

  //TELL SLAVES WE ARE GOING TO MULTIPLY A*xvalue
  for(rank=1; rank <= instance->nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, instance->tag_ax, instance->comm);
    }

  //SEND xvalue TO THE SLAVES
  MPI_Bcast(xvalue, instance->rdA + 1, MPI_FLOAT, 0, instance->comm);

  //SLAVES MULTIPLY.....

  //ASSEMBLES SEGMENTS OF RESULT FROM SLAVES INTO FINAL RESULT
  MPI_Gatherv(dummyFloat, 0, MPI_FLOAT, result, counts, displacements, MPI_FLOAT, 0, instance->comm);

  free(counts); free(displacements);

  return;
}

//NOTE: xvalue must be of length ldA, result is of length rdA+1
extern void multiply_ATx(float* xvalue, float* result, ISTAinstance_mpi* instance)
{
  int rank;
  float* temp = calloc(instance->rdA+1, sizeof(float));
  int* counts = calloc(instance->nslaves + 1, sizeof(int));
  int* displacements = calloc(instance->nslaves + 1, sizeof(int));
  if(counts==NULL || displacements==NULL)
    fprintf(stdout,"Unable to allocate memory!");
  for(rank=1; rank <= instance->nslaves; rank++)
    {
      counts[rank] = instance->slave_ldA;
      displacements[rank] = instance->slave_ldA*(rank-1);
    }

  //Tell slaves we are going to multiply A' * xvalue
  for(rank=1; rank <= instance->nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, instance->tag_atx, instance->comm);
    }

  //Split xvalue into subvectors and distribute among the slaves
  MPI_Scatterv(xvalue, counts, displacements, MPI_FLOAT, temp, 0, MPI_FLOAT, 0, instance->comm);

  //Slaves multiply....

  //Sum up the results of each slave multiplication
  MPI_Reduce(temp, result, instance->rdA + 1, MPI_FLOAT, MPI_SUM, 0, instance->comm);

  free(temp); free(counts); free(displacements);

  return;
}

//NOTE: xvalue and result are of length rdA+1  
extern void multiply_ATAx(float* xvalue, float* result, ISTAinstance_mpi* instance)
{
  int rank;
  float* temp = calloc(instance->rdA + 1, sizeof(float));
  if(temp==NULL)
    fprintf(stdout,"Unable to allocate memory!");


  //Tell slaves we are going to do A^t*A*xvalue
  for(rank=1; rank <= instance->nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, instance->tag_atax, instance->comm);
    }

  //Send xvalue to the slaves
  MPI_Bcast(xvalue, instance->rdA + 1, MPI_FLOAT, 0, instance->comm);

  //Slaves multiply...

  //Sum up the results of each slave multiplication
  MPI_Reduce(temp, result, instance->rdA + 1, MPI_FLOAT, MPI_SUM, 0, instance->comm);

  free(temp);

  return;
}

extern void get_dat_matrix(float* A, int ldA, int rdA, int myrank, 
			   char* filename, int interceptFlag)
{
  int numRowsSkip = (myrank-1)*ldA;

  //Open the file
  FILE *matrixfile;
  matrixfile = fopen(filename, "r");
  if(matrixfile == NULL)
    fprintf(stderr, "File Open Failed on process %d!\n", myrank);

  //Skip to appropriate place in file
  char c;
  int count=0;
  while( count < numRowsSkip) {
    do {
      c = getc(matrixfile);
    } while( c != '\n');
    count++;
  }

  //READ IN MATRIX DATA, ADDING A FINAL COLUMN THAT IS ALL-ONE IF INTERCEPTFLAG IS TRUE
  //AND ALL-ZERO IF INTERCEPTFLAG IS FALSE
  float value, intercept;
  int row, col;

  if(interceptFlag)
    intercept=1.0;
  else
    intercept=0.0;

  //fprintf(stdout, "\nProcess %d getting matrix entries\n", myrank);
  for(row=0; row<ldA; row++) {
    for(col=0; col<rdA; col++) {
      fscanf(matrixfile, " %32f , ", &value);
      A[row*(rdA+1) + col] = value;
      //fscanf(matrixfile, " , ");
    }
    A[row*(rdA+1) + rdA] = intercept;
  }	 

  fclose(matrixfile);

  //int i;

  //  fprintf(stdout,"Slave %d getting %d by %d matrix\n", myrank, ldA, rdA);

  //for( i=0; i<ldA*rdA; i++)
  //{
  //  //A[i] = myrank * (i+1) * 0.1;
  //  A[i] =  (float)rand()/(1.0 * (float)RAND_MAX);
  //}

  return;
}
