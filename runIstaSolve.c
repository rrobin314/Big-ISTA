#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cblas.h>
#include"istalib.h"

static void getMasterParams(char* parameterFile, char* xfilename, char* bfilename, char* Matrixfilename, 
			    int* ldA, int* rdA, 
			    int* numLambdas, float* lambdaStart, float* lambdaFinish, 
			    float* gamma, float* step, char* regType, int* accel, 
			    int* MAX_ITER, float* MIN_XDIFF, float* MIN_FUNCDIFF);

static void getMatrix(float* A, int ldA, int rdA, char* Afilename);
static void getVector(float* b, int lengthb, char* bfilename);

#define MAX_FILENAME_SIZE 32

int main(int argc, char **argv)
{
  //srand(time(NULL));
  int i, j, ldA, rdA, accel, MAX_ITER, numLambdas;
  float lambdaStart, lambdaFinish, gamma, step, MIN_XDIFF, MIN_FUNCDIFF;
  char regType;
  char* xfilename = malloc(MAX_FILENAME_SIZE*sizeof(float));
  char* bfilename = malloc(MAX_FILENAME_SIZE*sizeof(float));
  char* Matrixfilename = malloc(MAX_FILENAME_SIZE*sizeof(float));

  //GET PARAMETERS FROM TXT FILE
  getMasterParams(argv[1], xfilename, bfilename, Matrixfilename, &ldA, &rdA,
		  &numLambdas, &lambdaStart, &lambdaFinish,
		  &gamma, &step, &regType, &accel,
		  &MAX_ITER, &MIN_XDIFF, &MIN_FUNCDIFF);


  //ALLOCATE MEMORY
  float *A = malloc(ldA*rdA*sizeof(float));
  float *b = malloc(ldA*sizeof(float));
  float *x0 = calloc(rdA, sizeof(float)); //Currently, we always intialize x0 to zero
  float *result = malloc(ldA*sizeof(float));
  if(A==NULL || b==NULL || x0==NULL || result==NULL)
    fprintf(stderr, "Memory Allocation Failed!");

  //READ IN A AND b FROM FILE
  fprintf(stderr, "filenames are: %s and %s and %s\n", xfilename, Matrixfilename, bfilename);
  getMatrix(A, ldA, rdA, Matrixfilename);
  getVector(b, ldA, bfilename);

  //PRINT INPUTS
  /*  fprintf(stdout, "Here's x:\n");
  for(i=0; i < rdA; i++)
    {
      fprintf(stdout, "%f ", x0[i]);
    }
  fprintf(stdout, "\n and here's b:\n");
  for(i=0; i < ldA; i++)
    {
      fprintf(stdout, "%f ", b[i]);
      }*/


  //Initialize ISTAinstance
  ISTAinstance* instance = ISTAinstance_new(A, ldA, rdA, b, lambdaStart, 
					    gamma, accel, regType, x0, step);

  //RUN ISTA
  for(j=0; j < numLambdas; j++) {
    if(numLambdas > 1)
      instance->lambda = lambdaStart - j * (lambdaStart - lambdaFinish) / (numLambdas - 1);

    ISTAsolve_lite(instance, MAX_ITER, MIN_XDIFF, MIN_FUNCDIFF);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 
		1.0, instance->A, instance->rdA, instance->xcurrent, 1, 0.0, result, 1);

    //print results
    /*    fprintf(stdout, "Here's the optimized x for lambda %f:\n", instance->lambda);
    for(i=0; i < rdA; i++)
      {
	fprintf(stdout, "%f ", instance->xcurrent[i]);
	}
    fprintf(stdout, "\n and here's the optimized A*x:\n");
    for(i=0; i < ldA; i++)
      {
	fprintf(stdout, "%f ", result[i]);
      }
    fprintf(stdout, "\n");
    }*/
  }


  //FREE MEMORY
  ISTAinstance_free(instance); free(result);
  free(xfilename); free(bfilename); free(Matrixfilename);

  return 0;
}


static void getMasterParams(char* parameterFile, char* xfilename, char* bfilename, char* Matrixfilename,
			    int* ldA, int* rdA, 
			    int* numLambdas, float* lambdaStart, float* lambdaFinish, 
			    float* gamma, float* step, char* regType, int* accel, 
			    int* MAX_ITER, float* MIN_XDIFF, float* MIN_FUNCDIFF) {
  FILE *paramFile;
  paramFile = fopen(parameterFile, "r");
  if(paramFile == NULL)
    fprintf(stderr, "ParamFile Open Failed!\n");

  //Read parameters:
  fscanf(paramFile, "FileNameForX0 : %31s", xfilename);
  fscanf(paramFile, " FileNameForB : %31s", bfilename);
  fscanf(paramFile, " FileNameForA : %31s", Matrixfilename);
  fscanf(paramFile, " numRows : %d", ldA);
  fscanf(paramFile, " numCols : %d", rdA);
  fscanf(paramFile, " numLambdas : %d %*128[^\n]", numLambdas);
  fscanf(paramFile, " lambdaStart : %16f", lambdaStart);
  fscanf(paramFile, " lambdaFinish : %16f", lambdaFinish);
  fscanf(paramFile, " StepSizeDecretion : %16f", gamma);
  fscanf(paramFile, " InitialStep : %16f", step);
  fscanf(paramFile, " RegressionType : %4s", regType);
  fscanf(paramFile, " FistaAcceleration : %d", accel);
  fscanf(paramFile, " MaximumIterations : %d", MAX_ITER);
  fscanf(paramFile, " MinimumXDelta : %16f", MIN_XDIFF);
  fscanf(paramFile, " MinimumFuncDelta : %16f", MIN_FUNCDIFF);

  fclose(paramFile);
  return;
}

static void getMatrix(float* A, int ldA, int rdA, char* Afilename) {
  FILE *paramFile;
  paramFile = fopen(Afilename, "r");
  if(paramFile == NULL)
    fprintf(stderr, "MatrixFile Open Failed!\n");

  int i;
  float value;
  for(i=0; i<rdA*ldA; i++) {
    fscanf(paramFile, " %32f , ", &value);
    A[i] = value;
  }

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
