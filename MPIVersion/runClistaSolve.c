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

int main(int argc, char **argv)
{
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
  int rank, i;
  int small_ldA=2;
  int total_ldA=nslaves*small_ldA;
  int rdA=4;

  float *xvalue, *result;
  xvalue = malloc(rdA*sizeof(float));
  result = malloc((total_ldA+rdA)*sizeof(float));
  if(xvalue==NULL || result==NULL)
    fprintf(stdout,"Unable to allocate memory!");
  for(i=0; i < rdA; i++)
    {
      xvalue[i] = (i+3) * 0.5;
    }

  //Do work!
  multiply_Ax(xvalue, rdA, small_ldA, result, nslaves, MPI_COMM_WORLD, TAG_AX);

  //print results
  fprintf(stdout, "Here's x:\n");
  for(i=0; i < rdA; i++)
    {
      fprintf(stdout, "%f ", xvalue[i]);
    }
  fprintf(stdout, "\nHere's A*x result:\n");
  for(i=0; i < total_ldA; i++)
    {
      fprintf(stdout, "%f ", result[i]);
    }

  //Do more work!
  multiply_ATAx(xvalue, rdA, result, nslaves, MPI_COMM_WORLD, TAG_ATAX);

  //print results
  fprintf(stdout, "\nHere's A'*A*x result:\n");
  for(i=0; i < rdA; i++)
    {
      fprintf(stdout, "%f ", result[i]);
    }

  fprintf(stdout, "\nClosing the program\n");

  free(xvalue); free(result);

  //Close the slave processes
  for(rank=1; rank <= nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, TAG_DIE, MPI_COMM_WORLD);
    }
}

static void slave(int myrank)
{
  int dummyInt;
  MPI_Status status;
  float *A, *xvalue, *resultVector, *tempHolder, *dummyFloat;
  int ldA=2;
  int rdA=4;

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
  fprintf(stdout,"Slave %d getting matrix A\n", myrank);
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

  free(A); free(xvalue); free(resultVector);
  return;
}
