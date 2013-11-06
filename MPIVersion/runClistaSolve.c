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
#define MAX_FILENAME_SIZE 32

static void master(int nslaves, char* parameterFile);
static void slave(int myrank, char* parameterFile);
static void getSlaveParams(char* parameterFile, int* ldA, int* rdA, char* matrixfilename);
static void getMasterParams(char* parameterFile, char* xfilename, char* bfilename, 
			    int* slave_ldA, int* rdA, 
			    int* numLambdas, float* lambdaStart, float* lambdaFinish, 
			    float* gamma, float* step, char* regType, int* accel, 
			    int* MAX_ITER, float* MIN_FUNCDIFF);
static void getVector(float* b, int lengthb, char* bfilename);

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
      master(nslaves, argv[1]);
    }
  else
    {
      slave(myrank, argv[2]);
    }

  MPI_Finalize();
  return 0;
}

static void master(int nslaves, char* parameterFile)
{
  //VARIABLE DECLARATIONS
  int rank, i, j, accel, MAX_ITER, slave_ldA, total_ldA, rdA, numLambdas;
  ISTAinstance_mpi* instance;
  float *xvalue, *result, *b, lambdaStart, lambdaFinish, gamma, step, MIN_FUNCDIFF;
  char regType, xfilename[MAX_FILENAME_SIZE], bfilename[MAX_FILENAME_SIZE];

  //GET VALUES FROM PARAMETER FILE
  getMasterParams(parameterFile, xfilename, bfilename, &slave_ldA, &rdA, 
		  &numLambdas, &lambdaStart, &lambdaFinish, &gamma, &step, &regType, &accel, 
		  &MAX_ITER, &MIN_FUNCDIFF);
  total_ldA = nslaves*slave_ldA;


  //ALLOCATE MEMORY
  xvalue = calloc(rdA,sizeof(float));
  result = malloc((total_ldA+rdA)*sizeof(float));
  b      = malloc((total_ldA)*sizeof(float));
  if(xvalue==NULL || result==NULL || b==NULL)
    fprintf(stdout,"Unable to allocate memory!");
  

  //ASSIGN VALUES TO XVALUE AND B
  //For now we just assign x0 to be all zeros (see calloc above)  
  getVector(b, total_ldA, bfilename);
  //getVector(xvalue, rdA, xfilename);
  
  //PRINT INPUTS
  /*  fprintf(stdout, "Here's x:\n");
  for(i=0; i < rdA; i++)
    {
      fprintf(stdout, "%f ", xvalue[i]);
    }
  fprintf(stdout, "\n and here's b:\n");
  for(i=0; i < total_ldA; i++)
    {
      fprintf(stdout, "%f ", b[i]);
    }
  */

  //CREATE ISTA OBJECT
  instance = ISTAinstance_mpi_new(slave_ldA, rdA, b, lambdaStart, gamma, 
				  accel, regType, xvalue, step,
				  nslaves, MPI_COMM_WORLD,
				  TAG_AX, TAG_ATX, TAG_ATAX, TAG_DIE);
  

  //RUN ISTA
  for(j=0; j < numLambdas; j++) {
    if(numLambdas > 1) {
      instance->lambda = lambdaStart * exp( log(lambdaFinish / lambdaStart) * j / (numLambdas - 1) );
      //instance->lambda = lambdaStart - j * (lambdaStart - lambdaFinish) / (numLambdas - 1);
    }

    ISTAsolve_lite(instance, MAX_ITER, MIN_FUNCDIFF);
    //multiply_Ax(xvalue, rdA, slave_ldA, result, nslaves, MPI_COMM_WORLD, TAG_AX);
    //multiply_ATx(yvalue, total_ldA, slave_ldA, rdA, result, nslaves, MPI_COMM_WORLD, TAG_ATX);


    //print results
    /*fprintf(stdout, "Here's the optimized x for lambda %f:\n", instance->lambda);
        for(i=0; i < rdA; i++)
      {
	fprintf(stdout, "%f ", xvalue[i]);
	}
    fprintf(stdout, "\n and here's the optimized A*x:\n");
    for(i=0; i < total_ldA; i++)
      {
	fprintf(stdout, "%f ", result[i]);
	}*/
    fprintf(stdout, "\n");
  }



  //CLOSE THE SLAVE PROCESSES AND FREE MEMORY
  fprintf(stdout, "\nClosing the program\n");
  for(rank=1; rank <= nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, TAG_DIE, MPI_COMM_WORLD);
    }

  free(result); ISTAinstance_mpi_free(instance);
  return;
}




static void slave(int myrank, char* parameterFile)
{
  int dummyInt, ldA, rdA;
  MPI_Status status;
  float *A, *xvalue, *resultVector, *tempHolder, *dummyFloat;
  char matrixfilename[MAX_FILENAME_SIZE];

  //GET PARAMETERS FROM THE TEXT FILE
  getSlaveParams(parameterFile, &ldA, &rdA, matrixfilename);

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
  get_dat_matrix(A, ldA, rdA, myrank, matrixfilename);
  fprintf(stdout,"A[0] and A[last] for slave %d is %f and %f \n", myrank, A[0], A[ldA*rdA-1]);

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


static void getMasterParams(char* parameterFile, char* xfilename, char* bfilename, 
			    int* slave_ldA, int* rdA, 
			    int* numLambdas, float* lambdaStart, float* lambdaFinish, 
			    float* gamma, float* step, char* regType, int* accel, 
			    int* MAX_ITER, float* MIN_FUNCDIFF) {
  FILE *paramFile;
  paramFile = fopen(parameterFile, "r");
  if(paramFile == NULL)
    fprintf(stderr, "ParamFile Open Failed!\n");

  //Read parameters:
  fscanf(paramFile, "FileNameForX0 : %32s", xfilename);
  fscanf(paramFile, " FileNameForB : %32s", bfilename);
  fscanf(paramFile, " numRowsForSlave : %d", slave_ldA);
  fscanf(paramFile, " numCols : %d", rdA);
  fscanf(paramFile, " numLambdas : %d %*128[^\n]", numLambdas);
  fscanf(paramFile, " lambdaStart : %16f", lambdaStart);
  fscanf(paramFile, " lambdaFinish : %16f", lambdaFinish);
  fscanf(paramFile, " StepSizeDecretion : %16f", gamma);
  fscanf(paramFile, " InitialStep : %16f", step);
  fscanf(paramFile, " RegressionType : %4s", regType);
  fscanf(paramFile, " FistaAcceleration : %d", accel);
  fscanf(paramFile, " MaximumIterations : %d", MAX_ITER);
  fscanf(paramFile, " MinimumFuncDelta : %16f", MIN_FUNCDIFF);

  fclose(paramFile);
  return;
}


static void getSlaveParams(char* parameterFile, int* ldA, int* rdA, char* matrixfilename) {

  FILE *paramFile;
  paramFile = fopen(parameterFile, "r");
  if(paramFile == NULL)
    fprintf(stderr, "ParamFile Open Failed!\n");

  fscanf(paramFile, "MatrixFileName : %32s", matrixfilename);
  fscanf(paramFile, " numRows : %d", ldA);
  fscanf(paramFile, " numCols : %d", rdA);

  fclose(paramFile);
  return;
}

static void getVector(float* b, int lengthb, char* bfilename) {
  FILE *paramFile;
  paramFile = fopen(bfilename, "r");
  if(paramFile == NULL)
    fprintf(stderr, "VectorFile Open Failed!\n");

  int i;
  float value;
  for(i=0; i<lengthb; i++) {
    fscanf(paramFile, " %32f , ", &value);
    b[i] = value;
  }

  fclose(paramFile);
  return;
}
