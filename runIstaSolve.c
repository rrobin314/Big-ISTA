#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<cblas.h>
#include"istalib.h"

static void getMasterParams(char* parameterFile, char* xfilename, char* bfilename, char* Matrixfilename, 
			    char* outfilename, int* ldA, int* rdA, 
			    int* numLambdas, float* lambdaStart, float* lambdaFinish, 
			    float* gamma, float* step, char* regType, int* accel, 
			    int* MAX_ITER, float* MIN_FUNCDIFF);

static void getMatrix(float* A, int ldA, int rdA, char* Afilename);
static void getVector(float* b, int lengthb, char* bfilename);
static void writeResults(ISTAinstance* instance, char* outfilename, 
			 char* Matrixfilename, char* bfilename, float finalLambda);

#define MAX_FILENAME_SIZE 64

int main(int argc, char **argv)
{
  //srand(time(NULL));
  int i, j, ldA, rdA, accel, MAX_ITER, numLambdas;
  float lambdaStart, lambdaFinish, gamma, step, MIN_FUNCDIFF;
  char regType;
  char* xfilename = malloc(MAX_FILENAME_SIZE*sizeof(float));
  char* bfilename = malloc(MAX_FILENAME_SIZE*sizeof(float));
  char* Matrixfilename = malloc(MAX_FILENAME_SIZE*sizeof(float));
  char* outfilename = malloc(MAX_FILENAME_SIZE*sizeof(float));

  //GET PARAMETERS FROM TXT FILE
  getMasterParams(argv[1], xfilename, bfilename, Matrixfilename, outfilename, &ldA, &rdA,
		  &numLambdas, &lambdaStart, &lambdaFinish,
		  &gamma, &step, &regType, &accel,
		  &MAX_ITER, &MIN_FUNCDIFF);


  //ALLOCATE MEMORY
  float *A = malloc(ldA*rdA*sizeof(float));
  float *b = malloc(ldA*sizeof(float));
  float *x0 = calloc(rdA, sizeof(float)); //Currently, we always intialize x0 to zero
  float *lambdas = malloc(numLambdas*sizeof(float));
  float *result = malloc((ldA+rdA)*sizeof(float));
  if(A==NULL || b==NULL || x0==NULL || result==NULL || lambdas==NULL)
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

  //CALCULATE LAMBDA PATH
  calcLambdas(lambdas, numLambdas, lambdaStart, lambdaFinish, A, ldA, rdA, b, result);

  //Initialize ISTAinstance
  ISTAinstance* instance = ISTAinstance_new(A, ldA, rdA, b, lambdas[0], 
					    gamma, accel, regType, x0, step);

  //RUN ISTA
  for(j=0; j < numLambdas; j++) {
    if(j > 0)
      instance->lambda = lambdas[j];

    ISTAsolve_lite(instance, MAX_ITER, MIN_FUNCDIFF);
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
	}*/
    fprintf(stdout, "\n");
  }

  //WRITE RESULTS TO FILE:
  writeResults(instance, outfilename, Matrixfilename, bfilename, lambdas[numLambdas-1]);

  


  //FREE MEMORY
  ISTAinstance_free(instance); free(result); free(lambdas);
  free(xfilename); free(bfilename); free(Matrixfilename);

  return 0;
}


static void getMasterParams(char* parameterFile, char* xfilename, char* bfilename, char* Matrixfilename,
			    char* outfilename, int* ldA, int* rdA, 
			    int* numLambdas, float* lambdaStart, float* lambdaFinish, 
			    float* gamma, float* step, char* regType, int* accel, 
			    int* MAX_ITER, float* MIN_FUNCDIFF) {
  FILE *paramFile;
  paramFile = fopen(parameterFile, "r");
  if(paramFile == NULL)
    fprintf(stderr, "ParamFile Open Failed!\n");

  //Read parameters:
  fscanf(paramFile, "FileNameForX0 : %63s", xfilename);
  fscanf(paramFile, " FileNameForB : %63s", bfilename);
  fscanf(paramFile, " FileNameForA : %63s", Matrixfilename);
  fscanf(paramFile, " OutputFile : %63s", outfilename);
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

static void writeResults(ISTAinstance* instance, char* outfilename, 
			 char* Matrixfilename, char* bfilename, float finalLambda) {
  char regForm[10] = "linear";
  char accelForm[10] = "FISTA";
  int i;

  if(instance->regressionType == 'o')
    strcpy(regForm, "logistic");
  if(instance->acceleration == 0)
    strcpy(accelForm, "ISTA");
  
  FILE* outFILE;
  outFILE = fopen(outfilename,"w");
  if(outFILE!=NULL) {
    fprintf(outFILE, "Results for %s regression using %s algorithm. \n", regForm, accelForm);
    fprintf(outFILE, "Using data from:\nMatrix File %s \nVector File %s \n", Matrixfilename, bfilename);
    fprintf(outFILE, "and final regularization weight %f \n\nFINAL X VECTOR:\n", finalLambda);
    for(i=0; i < instance->rdA; i++) {
      fprintf(outFILE, "%f \n", instance->xcurrent[i]);
    }
    fclose(outFILE);
  }
}
