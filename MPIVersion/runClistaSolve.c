#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<cblas.h>
#include"mpi.h"
#include"clistaLib.h"

#define TAG_AX 267
#define TAG_ATX 832
#define TAG_DIE 451
#define TAG_ATAX 674

static void master(int nslaves);
static void slave(int myrank);

//Time wasn't working well to seed the random number generator.
//This function counts the number of cycles since the processor was
//turned on.  We use this to seed rand()
unsigned long long rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)hi << 32) | lo;
}

int main(int argc, char **argv)
{
  srand(rdtsc());
  int myrank, nslaves;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  //Get number of slaves by getting total number and subtracting 1
  MPI_Comm_size(MPI_COMM_WORLD, &nslaves);
  nslaves = nslaves - 1;

  if (myrank==0)
    {
      master(nslaves);
    }
  else
    {
      slave(myrank);
    }

  MPI_Finalize();
  return 0;
}

static void master(int nslaves)
{
  int rank, i, accel, MAX_ITER;
  int small_ldA=2;
  int total_ldA=nslaves*small_ldA;
  int rdA=15;
  ISTAinstance_mpi* instance;
  float *xvalue, *yvalue, *result, *b, lambda, gamma, step, MIN_XDIFF, MIN_FUNCDIFF;
  char regType;

  //Initialize values:
  lambda = 0.1;
  gamma = 0.9;
  step = 1.0;
  regType = 'l';
  accel = 0;
  MAX_ITER=10000;
  MIN_XDIFF=0.0001;
  MIN_FUNCDIFF=0.00001;

  //Allocate Memory
  xvalue = malloc(rdA*sizeof(float));
  yvalue = malloc(total_ldA*sizeof(float));
  result = malloc((total_ldA+rdA)*sizeof(float));
  b      = malloc((total_ldA)*sizeof(float));
  if(xvalue==NULL || result==NULL)
    fprintf(stdout,"Unable to allocate memory!");
  
  //Assign values to xvalue and b
  for(i=0; i < rdA; i++)
    {
      xvalue[i] = (i+3) * 0.5;
    }
  for(i=0; i < total_ldA; i++)
    {
      b[i] =  (float)rand()/(1.0 * (float)RAND_MAX);
    }

  fprintf(stdout, "Here's x:\n");
  for(i=0; i < rdA; i++)
    {
      fprintf(stdout, "%f ", xvalue[i]);
    }
  fprintf(stdout, "\n and here's b:\n");
  for(i=0; i < total_ldA; i++)
    {
      fprintf(stdout, "%f ", b[i]);
    }
  
  //yvalue[0] = 1.0; yvalue[1] = 0.0; yvalue[2] = 0.0; 
  //yvalue[3] = 1.0; yvalue[4] = 2.0; yvalue[5] = 0.0; 
  //b[0]=0.5; b[1]=6.1; b[2]=5.0; b[3]=11.2; b[4]=7.5; b[5]=18.3;
  
  //Create ISTA object
  instance = ISTAinstance_mpi_new(small_ldA, rdA, b, lambda, gamma, 
				  accel, regType, xvalue, step,
				  nslaves, MPI_COMM_WORLD,
				  TAG_AX, TAG_ATX, TAG_ATAX, TAG_DIE);
  

  //Do work!
  ISTAsolve_lite(instance, MAX_ITER, MIN_XDIFF, MIN_FUNCDIFF);
  multiply_Ax(xvalue, rdA, small_ldA, result, nslaves, MPI_COMM_WORLD, TAG_AX);
  //multiply_ATx(yvalue, total_ldA, small_ldA, rdA, result, nslaves, MPI_COMM_WORLD, TAG_ATX);

  //print results
  fprintf(stdout, "Here's the optimized x:\n");
  for(i=0; i < rdA; i++)
    {
      fprintf(stdout, "%f ", xvalue[i]);
    }
  fprintf(stdout, "\n and here's the optimized A*x:\n");
  for(i=0; i < total_ldA; i++)
    {
      fprintf(stdout, "%f ", result[i]);
    }



  /*
  fprintf(stdout, "\nHere's A*x result:\n");
  for(i=0; i < total_ldA; i++)
    {
      fprintf(stdout, "%f ", result[i]);
    }

  //Do more work!
  //multiply_ATAx(xvalue, rdA, result, nslaves, MPI_COMM_WORLD, TAG_ATAX);
  //result[0] = ISTAloss_func_mpi(xvalue, instance);
  //fprintf(stdout, "\nHere's the loss: %f \n", result[0]);
  ISTAgrad(instance);
  fprintf(stdout, "\nHere's grad of x:\n");
  for(i=0; i < rdA; i++)
    {
      fprintf(stdout, "%f ", (instance->gradvalue)[i]);
    }


  //print results
  //  fprintf(stdout, "\nHere's A'*A*x result:\n");
  //  for(i=0; i < rdA; i++)
  //    {
  //     fprintf(stdout, "%f ", result[i]);
  //    }
  */




  fprintf(stdout, "\nClosing the program\n");

  //CLOSE THE SLAVE PROCESSES AND FREE MEMORY
  for(rank=1; rank <= nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, TAG_DIE, MPI_COMM_WORLD);
    }

  free(result); ISTAinstance_mpi_free(instance);

}

static void slave(int myrank)
{
  int dummyInt;
  MPI_Status status;
  float *A, *xvalue, *resultVector, *tempHolder, *dummyFloat;
  int ldA=2;
  int rdA=15;

  //ALLOCATE A, TEMPHOLDER, RESULTVECTOR and XVALUE
  A = (float*)calloc(ldA*rdA, sizeof(float));
  if(A==NULL)
    fprintf(stdout,"Unable to allocate memory!");

  xvalue = malloc( (ldA+rdA)*sizeof(float) );
  tempHolder = malloc( (ldA+rdA)*sizeof(float) ); //place holder for intermediate calculations
  resultVector = malloc( (ldA+rdA)*sizeof(float) );
  if(xvalue==NULL || tempHolder==NULL || resultVector==NULL)
    fprintf(stdout,"Unable to allocate memory!");


  //FILL A WITH DESIRED VALUES
  get_dat_matrix(A, ldA, rdA, myrank);
  fprintf(stdout,"A[0] for slave %d is %f \n", myrank, A[0]);

  //COMPUTATION LOOP
  while(1)
    {
      MPI_Recv(&dummyInt, 0, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      //Check the tag to determine what to do next
      if (status.MPI_TAG == TAG_DIE)
	{
	  break;
	}

      else if (status.MPI_TAG == TAG_AX)
	{
	  //Multiply A * x

	  //Get xvalue
	  MPI_Bcast(xvalue, rdA, MPI_FLOAT, 0, MPI_COMM_WORLD);

	  //Multiply: resultVector = A*xvalue
	  cblas_sgemv(CblasRowMajor, CblasNoTrans, ldA, rdA, 1.0, A, rdA, 
		      xvalue, 1, 0.0, resultVector, 1);

	  //Gather xvalues
	  MPI_Gatherv(resultVector, ldA, MPI_FLOAT, dummyFloat, &dummyInt, &dummyInt, MPI_FLOAT, 0, MPI_COMM_WORLD);

	  
	}

      else if (status.MPI_TAG == TAG_ATX)
	{
	  //Multiply A^t * x
	  
	  //Get xvalue
	  MPI_Scatterv(dummyFloat, &dummyInt, &dummyInt, MPI_FLOAT, 
		       xvalue, ldA, MPI_FLOAT, 0, MPI_COMM_WORLD);

	  //Multiply: resultVector = A'*xvalue
	  cblas_sgemv(CblasRowMajor, CblasTrans, ldA, rdA, 1.0, A, rdA, 
		      xvalue, 1, 0.0, resultVector, 1);

	  //Sum resultVectors to get final result
	  MPI_Reduce(resultVector, dummyFloat, rdA, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);


	}
      else if (status.MPI_TAG == TAG_ATAX)
	{
	  //Multiply A^t * A * x

	  //Get xvalue
	  MPI_Bcast(xvalue, rdA, MPI_FLOAT, 0, MPI_COMM_WORLD);

	  //Multiply: tempHolder = A*xvalue
	  cblas_sgemv(CblasRowMajor, CblasNoTrans, ldA, rdA, 1.0, A, rdA, 
		      xvalue, 1, 0.0, tempHolder, 1);
	  //Multiply: resultVector = A^t * tempHolder
	  cblas_sgemv(CblasRowMajor, CblasTrans, ldA, rdA, 1.0, A, rdA,
		      tempHolder, 1, 0.0, resultVector, 1);

	  //Gather and sum results
	  MPI_Reduce(resultVector, dummyFloat, rdA, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	  
	}

    }

  free(A); free(xvalue); free(tempHolder); free(resultVector);
  return;
}
