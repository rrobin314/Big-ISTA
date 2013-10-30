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

  instance->xprevious = malloc(rdA*sizeof(float));
  if ( instance->xprevious==NULL )
    fprintf(stdout, "Unable to allocate memory\n");

  instance->searchPoint = malloc(rdA*sizeof(float));
  if ( instance->searchPoint==NULL )
    fprintf(stdout, "Unable to allocate memory\n");
  cblas_scopy(rdA, instance->xcurrent, 1, instance->searchPoint, 1);

  instance->gradvalue = malloc(rdA*sizeof(float));
  instance->eta = malloc((instance->ldA + rdA)*sizeof(float));
  if ( instance->gradvalue==NULL || instance->eta==NULL )
    fprintf(stdout, "Unable to allocate memory\n");

  fprintf(stdout,"Created ISTA instance with parameters:\n nslaves: %d slave_ldA: %d ldA: %d rdA: %d \n lambda: %f gamma: %f accel: %d regType: %c step: %f  \n b[0]: %f b[last]: %f \n x[0]: %f x[last]: %f \n", nslaves, slave_ldA, instance->ldA, rdA, lambda, gamma, acceleration,
	  regressionType, step, b[0], b[instance->ldA-1], xvalue[0], xvalue[rdA-1]);

  return instance;
}


void ISTAinstance_mpi_free(ISTAinstance_mpi* instance)
{
  // Frees an entire ISTAinstance pointer
  free(instance->eta); 
  free(instance->gradvalue); 
  free(instance->searchPoint); 
  free(instance->xprevious);
  free(instance->xcurrent);
  free(instance->stepsize); 
  free(instance-> b);
  free(instance);
}

/*

void ISTAsolve(float* A, int ldA, int rdA, float* b, float lambda, float gamma, 
	       int acceleration, char regressionType, float* xvalue, 
	       int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF)
{
  //This method updates xvalue to reflect the solution to the optimization

  //INITIALIZATION
  ISTAinstance* instance = malloc(sizeof(ISTAinstance));
  if ( instance==NULL )
    fprintf(stdout, "Unable to allocate memory\n");

  instance->A = A;
  instance->ldA = ldA;
  instance->rdA = rdA;
  instance->b = b;
  instance->lambda = lambda;
  instance->gamma = gamma;
  instance->acceleration = acceleration;
  instance->regressionType = regressionType;
  instance->xcurrent = xvalue;

  //Allocate memory for values used during the calculation
  instance->stepsize = malloc(sizeof(float));
  *(instance->stepsize) = 1.0;

  instance->xprevious = malloc(rdA*sizeof(float));
  if ( instance->xprevious==NULL )
    fprintf(stdout, "Unable to allocate memory\n");

  instance->searchPoint = malloc(rdA*sizeof(float));
  if ( instance->searchPoint==NULL )
    fprintf(stdout, "Unable to allocate memory\n");
  cblas_scopy(rdA, instance->xcurrent, 1, instance->searchPoint, 1);

  instance->gradvalue = malloc(rdA*sizeof(float));
  instance->eta = malloc((ldA+rdA)*sizeof(float));
  if ( instance->gradvalue==NULL || instance->eta==NULL )
    fprintf(stdout, "Unable to allocate memory\n");

  //Initialize stop values:
  int iter=0;
  float xdiff=1;
  float funcdiff=1;

  while(iter < MAX_ITER && xdiff > MIN_XDIFF && funcdiff > MIN_FUNCDIFF)
    {
      cblas_scopy(rdA, instance->xcurrent, 1, instance->xprevious, 1); //set xprevious to xcurrent

      //RUN BACKTRACKING ROUTINE
       ISTAbacktrack( instance );

      //UPDATE TERMINATING VALUES
      cblas_saxpy(rdA, -1.0, instance->xcurrent, 1, instance->xprevious, 1); //xprevious now holds "xprevious - xcurrent"
      xdiff = cblas_snrm2(rdA, instance->xprevious, 1);

      funcdiff = ISTAloss_func_mpi(instance->searchPoint, instance) - ISTAloss_func_mpi(instance->xcurrent, instance);
      funcdiff += instance->lambda * cblas_sasum(rdA, instance->searchPoint, 1);
      funcdiff -= instance->lambda * cblas_sasum(rdA, instance->xcurrent, 1);

      //UPDATE SEARCHPOINT
      if( instance->acceleration ) //FISTA searchpoint
	{
	  cblas_sscal(rdA, - iter / (float)(iter + 2), instance->xprevious, 1);
	  cblas_saxpy(rdA, 1.0, instance->xcurrent, 1, instance->xprevious, 1); //now xprevious equals what we want
	  cblas_scopy(rdA, instance->xprevious, 1, instance->searchPoint, 1);
	}
      else //regular ISTA searchpoint
	{
	  cblas_scopy(rdA, instance->xcurrent, 1, instance->searchPoint, 1);
	}

      //UPDATE ITERATOR
      iter++;
    }
  
  fprintf(stdout, "iter: %d xdiff: %f funcdiff: %f\n", iter, xdiff, funcdiff);
  fprintf(stdout, "final regression function value: %f\n", ISTAloss_func_mpi(instance->xcurrent, instance) );

  //FREE MEMORY
  free(instance->eta); free(instance->gradvalue); free(instance->searchPoint); free(instance->xprevious);
  free(instance->stepsize); free(instance);
									       
}

*/

void ISTAsolve_lite(ISTAinstance_mpi* instance, int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF )
{
  // This version of ISTAsolve solve does not allocate any memory

  //Initialize stop values:
  int iter=0;
  float xdiff=1;
  float funcdiff=1;

  fprintf(stdout, "intial regression function value for lambda %f: %f\n", instance->lambda, ISTAloss_func_mpi(instance->xcurrent, instance) );

  while(iter < MAX_ITER && xdiff > MIN_XDIFF && funcdiff > MIN_FUNCDIFF)
    {
      cblas_scopy(instance->rdA, instance->xcurrent, 1, instance->xprevious, 1); //set xprevious to xcurrent

      //RUN BACKTRACKING ROUTINE
      ISTAbacktrack( instance );

      //UPDATE TERMINATING VALUES
      cblas_saxpy(instance->rdA, -1.0, instance->xcurrent, 1, instance->xprevious, 1); //xprevious now holds "xprevious - xcurrent"
      xdiff = cblas_snrm2(instance->rdA, instance->xprevious, 1);

      funcdiff = ISTAloss_func_mpi(instance->searchPoint, instance) - ISTAloss_func_mpi(instance->xcurrent, instance);
      funcdiff += instance->lambda * cblas_sasum(instance->rdA, instance->searchPoint, 1);
      funcdiff -= instance->lambda * cblas_sasum(instance->rdA, instance->xcurrent, 1);

      //UPDATE SEARCHPOINT
      if( instance->acceleration ) //FISTA searchpoint
	{
	  cblas_sscal(instance->rdA, - iter / (float)(iter + 2), instance->xprevious, 1);
	  cblas_saxpy(instance->rdA, 1.0, instance->xcurrent, 1, instance->xprevious, 1); //now xprevious equals what we want
	  cblas_scopy(instance->rdA, instance->xprevious, 1, instance->searchPoint, 1);
	}
      else //regular ISTA searchpoint
	{
	  cblas_scopy(instance->rdA, instance->xcurrent, 1, instance->searchPoint, 1);
	}

      //UPDATE ITERATOR
      iter++;
    }

  fprintf(stdout, "\niter: %d xdiff: %f funcdiff: %f\n", iter, xdiff, funcdiff);
  fprintf(stdout, "final regression function value: %f\n", ISTAloss_func_mpi(instance->xcurrent, instance) );

}

/*

float** ISTAsolve_pathwise(float* lambdas, int num_lambdas, ISTAinstance* instance, 
			   int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF )
{
  // lambdas is an array of floats, ordered from largest to smallest, giving the lambdas we will solve for.
  // for each lambda in lambdas, we run ISTA and return the final xvalue

  float** values = malloc( num_lambdas*sizeof(float*) );
  if( values==NULL )
    fprintf(stdout, "Unable to allocate memory");

  int i;

  for( i=0; i < num_lambdas; i++)
    {
      instance->lambda = lambdas[i];

      // Solve with new lambda value
      ISTAsolve_lite( instance, MAX_ITER, MIN_XDIFF, MIN_FUNCDIFF );

      // record solution in values[i]
      values[i] = malloc( (instance->rdA) * sizeof(float) );
      if ( values[i] == NULL )
	fprintf(stdout, "Unable to allocate memory");
      cblas_scopy(instance->rdA, instance->xcurrent, 1, values[i], 1); 
    }

  return values;
}

float ISTAcrossval(ISTAinstance* instance, int* folds, int num_folds, 
		   int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF )
{
  float meanTotalError = 0.0;
  int i, j, iter, currentFold;
  float xdiff, funcdiff, foldError;

  // for each fold number, we run ISTA on the rows not included in that fold
  for( currentFold=0; currentFold < num_folds; currentFold++)
    {
      
      iter=0;
      xdiff = 1.0;
      funcdiff = 1.0;
      foldError = 0.0;

      int numRowsInFold = 0;
      for(i=0; i < instance->ldA; i++)
	{
	  if(currentFold == folds[i])
	    numRowsInFold++;
	}
      if(numRowsInFold == 0)
	fprintf(stdout, "no rows in this folds!");

      // reset xcurrent to 0 vector
      cblas_sscal(instance->rdA, 0.0, instance->xcurrent, 1);

      // Now run the usual ISTA solve code, but calling the crossval specific functions
      while(iter < MAX_ITER && xdiff > MIN_XDIFF && funcdiff > MIN_FUNCDIFF)
	{
	  cblas_scopy(instance->rdA, instance->xcurrent, 1, instance->xprevious, 1); //set xprevious to xcurrent
	  
	  //RUN BACKTRACKING ROUTINE
	  ISTAbacktrack_cv( instance, currentFold, folds);
	  
	  //UPDATE TERMINATING VALUES
	  cblas_saxpy(instance->rdA, -1.0, instance->xcurrent, 1, instance->xprevious, 1); //xprevious now holds "xprevious - xcurrent"
	  xdiff = cblas_snrm2(instance->rdA, instance->xprevious, 1);
	  
	  funcdiff = ISTAregress_func_cv(instance->searchPoint, instance, currentFold, folds, 0)
	              - ISTAregress_func_cv(instance->xcurrent, instance, currentFold, folds, 0);
	  funcdiff += instance->lambda * cblas_sasum(instance->rdA, instance->searchPoint, 1);
	  funcdiff -= instance->lambda * cblas_sasum(instance->rdA, instance->xcurrent, 1);
	  
	  //UPDATE SEARCHPOINT
	  if( instance->acceleration ) //FISTA searchpoint
	     {
	       cblas_sscal(instance->rdA, - iter / (float)(iter + 2), instance->xprevious, 1);
	       cblas_saxpy(instance->rdA, 1.0, instance->xcurrent, 1, instance->xprevious, 1); //now xprevious equals what we want
	       cblas_scopy(instance->rdA, instance->xprevious, 1, instance->searchPoint, 1);
	     }
	  else //regular ISTA searchpoint
	    {
	      cblas_scopy(instance->rdA, instance->xcurrent, 1, instance->searchPoint, 1);
	    }
	  
	  //UPDATE ITERATOR
	  iter++;
	} 
       

      // Now calculate the average test error on the rows in the fold
      if(numRowsInFold != 0)
	foldError = ISTAregress_func_cv(instance->xcurrent, instance, currentFold, folds, 1) / numRowsInFold;
      
      // Update meanTotalError
      meanTotalError += foldError;

    }

  meanTotalError = meanTotalError / num_folds;
  return meanTotalError;
}

*/

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
    cblas_scopy(instance->rdA, instance->searchPoint, 1, instance->xcurrent, 1);
    cblas_saxpy(instance->rdA, -(*(instance->stepsize)), instance->gradvalue, 1, instance->xcurrent, 1);
    soft_threshold(instance->xcurrent, instance->rdA, instance->lambda * (*(instance->stepsize)));

    //calculate difference that, when negative, guarantees the objective function decreases
    difference = ISTAloss_func_mpi(instance->xcurrent, instance) - 
	         ISTAloss_func_mpi(instance->searchPoint, instance);
    cblas_scopy(instance->rdA, instance->xcurrent, 1, instance->eta, 1);
    cblas_saxpy(instance->rdA, -1.0, instance->searchPoint, 1, instance->eta, 1); //eta now holds "xcurrent - searchpoint"
    difference -= cblas_sdot(instance->rdA, instance->eta, 1, instance->gradvalue, 1);
    difference -= cblas_sdot(instance->rdA, instance->eta, 1, instance->eta, 1) / (2 * (*(instance->stepsize)) );

    numTrials++;

  } while(numTrials < 100 && difference > 0);
  
  if(numTrials == 100)
    fprintf(stdout, "backtracking failed\n");

}

/*

void ISTAbacktrack_cv(ISTAinstance* instance, int currentFold, int* folds)
{
  // initialize 
  int i;
  int numTrials = 0;
  float difference;

  // calculate gradient at current searchPoint
  ISTAgrad_cv(instance, currentFold, folds);
  
  do
  {
    if(numTrials > 0) // dont update stepsize the first time through 
      *(instance->stepsize) *= instance->gamma;

    // update xcurrent = soft(  searchPoint - stepsize*gradvalue , lambda*stepsize )  
    cblas_scopy(instance->rdA, instance->searchPoint, 1, instance->xcurrent, 1);
    cblas_saxpy(instance->rdA, -(*(instance->stepsize)), instance->gradvalue, 1, instance->xcurrent, 1);
    soft_threshold(instance->xcurrent, instance->rdA, instance->lambda * (*(instance->stepsize)));

    // calculate difference that, when negative, guarantees the objective function decreases 
    difference = ISTAregress_func_cv(instance->xcurrent, instance, currentFold, folds, 0) - 
	         ISTAregress_func_cv(instance->searchPoint, instance, currentFold, folds, 0);
    cblas_scopy(instance->rdA, instance->xcurrent, 1, instance->eta, 1);
    cblas_saxpy(instance->rdA, -1.0, instance->searchPoint, 1, instance->eta, 1); //eta now holds "xcurrent - searchpoint"
    difference -= cblas_sdot(instance->rdA, instance->eta, 1, instance->gradvalue, 1);
    difference -= cblas_sdot(instance->rdA, instance->eta, 1, instance->eta, 1) / (2 * (*(instance->stepsize)) );

    numTrials++;

  } while(numTrials < 100 && difference > 0);
  
  if(numTrials == 100)
    fprintf(stdout, "backtracking failed\n");

}

*/

extern void ISTAgrad(ISTAinstance_mpi* instance)
{
  // THIS FUNCTION CALCULATES THE GRADIENT OF THE SMOOTH FUNCTION ISTAloss_func_mpi
  // AT THE POINT "searchPoint" and stores it in "gradvalue"

  int i;
  
  switch ( instance->regressionType )
    {
    case 'l': // Here we calculate the gradient: 2*A'*(A*searchPoint - b)  
      multiply_Ax(instance->searchPoint, instance->rdA, instance->slave_ldA, instance->eta, 
		  instance->nslaves, instance->comm, instance->tag_ax);

      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*searchPoint - b

      //NOTE: WE ARE MISSING THE SCALAR 2 IN THIS CALCULATION, DOES THIS MATTER?
      multiply_ATx(instance->eta, instance->ldA, instance->slave_ldA, instance->rdA,
		   instance->gradvalue, instance->nslaves, instance->comm, instance->tag_atx);

      break;
      
    case 'o': // Here we calculate the gradient:A'*(p(A*searchPoint) - b) where p is the logistic function 

      multiply_Ax(instance->searchPoint, instance->rdA, instance->slave_ldA, instance->eta, 
		  instance->nslaves, instance->comm, instance->tag_ax);

      for(i=0; i < instance->ldA; i++)
	{
	  instance->eta[i] = 1 / (1 + exp( -(instance->eta[i]) ) ) - instance->b[i];
	}

      multiply_ATx(instance->eta, instance->ldA, instance->slave_ldA, instance->rdA,
		   instance->gradvalue, instance->nslaves, instance->comm, instance->tag_atx);

      break;
      
    }

}

/*

void ISTAgrad_cv(ISTAinstance* instance, int currentFold, int* folds)
{
  // THIS FUNCTION CALCULATES THE GRADIENT OF THE SMOOTH FUNCTION ISTAregress_func
  // AT THE POINT "searchPoint" and stores it in "gradvalue" ONLY USING THE ROWS
  // OF A AND b CORRESPONDING TO THOSE NOT IN THE CURRENT FOLD

  int i;
  
  switch ( instance->regressionType )
    {
    case 'l': // Here we calculate the gradient: 2*A'*(A*searchPoint - b)  
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  instance->searchPoint, 1, 0.0, instance->eta, 1);
      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*searchPoint - b

      // kill those entries of eta that lie in the fold (test set)
      for(i=0; i < instance->ldA; i++)
	{
	  if(folds[i] == currentFold)
	    instance->eta[i] = 0.0;
	}

      cblas_sgemv(CblasRowMajor, CblasTrans, instance->ldA, instance->rdA, 2.0, instance->A, instance->rdA,
		  instance->eta, 1, 0.0, instance->gradvalue, 1);
      break;

    case 'o': // Here we calculate the gradient:A'*(p(A*searchPoint) - b) where p is the logistic function 
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  instance->searchPoint, 1, 0.0, instance->eta, 1);
      for(i=0; i < instance->ldA; i++)
	{
	  if(folds[i] == currentFold)
	    instance->eta[i] = 0.0;
	  else
	    instance->eta[i] = 1 / (1 + exp( -(instance->eta[i]) ) ) - instance->b[i];
	}

      cblas_sgemv(CblasRowMajor, CblasTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA,
		  instance->eta, 1, 0.0, instance->gradvalue, 1);
      break;

    }

}



*/






float ISTAloss_func_mpi(float* xvalue, ISTAinstance_mpi* instance)
{
  //THIS FUNCTION REPRESENTS THE SMOOTH FUNCTION THAT WE ARE TRYING TO OPTIMIZE
  //WHILE KEEPING THE REGULARIZATION FUNCTION (USUALLY THE 1-NORM) SMALL

  int i;
  float value = 0;

  switch (instance->regressionType) 
    {
    case 'l': //In this case, the regression function is ||A*xvalue - b||^2 
      multiply_Ax(xvalue, instance->rdA, instance->slave_ldA, instance->eta, 
		  instance->nslaves, instance->comm, instance->tag_ax);
      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*xvalue - b

      value = pow( cblas_snrm2(instance->ldA, instance->eta, 1 ), 2);
      break;

    case 'o': // Regression function: sum log(1+ e^(A_i * x)) - A_i * x * b_i 
      multiply_Ax(xvalue, instance->rdA, instance->slave_ldA, instance->eta, 
		  instance->nslaves, instance->comm, instance->tag_ax);
      for(i=0; i < instance->ldA; i++)
      	{
	  value += log( 1 + exp( instance->eta[i] )) - instance->eta[i] * instance->b[i];
      	}      
      break;
    }
      
  return value;
}


/*


float ISTAregress_func_cv(float* xvalue, ISTAinstance* instance, int currentFold, int* folds, int insideFold)
{
  // If insideFold==1, then we only consider those rows inside the fold
  // If insideFold==0, then we consider those rows outside the fold

  int j;
  float value = 0;

  switch (instance->regressionType) 
    {
    case 'l': //In this case, the regression function is ||A*xvalue - b||^2 
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  xvalue, 1, 0.0, instance->eta, 1);
      cblas_saxpy(instance->ldA, -1.0, instance->b, 1, instance->eta, 1); //eta now holds A*xvalue - b

      // Now kill all elements of eta that we don't want
      for(j=0; j < instance->ldA; j++)
	{
	  if( (folds[j] != currentFold && insideFold == 1) || (folds[j] == currentFold && insideFold == 0) )
	    instance->eta[j] = 0.0;
	}

      value = pow( cblas_snrm2(instance->ldA, instance->eta, 1 ), 2);
      break;

    case 'o': // Regression function: sum log(1+ e^(A_i * x)) - A_i * x * b_i 
      cblas_sgemv(CblasRowMajor, CblasNoTrans, instance->ldA, instance->rdA, 1.0, instance->A, instance->rdA, 
		  xvalue, 1, 0.0, instance->eta, 1);
      for(j=0; j < instance->ldA; j++)
	{
	  if( (folds[j] == currentFold && insideFold == 1) || (folds[j] != currentFold && insideFold == 0) )
	    value += log( 1 + exp( instance->eta[j] )) - instance->eta[j] * instance->b[j];
	}
      break;
    }
      
  return value;
}

*/

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

extern void multiply_Ax(float* xvalue, int lenx, int ldA, float* result, int nslaves, MPI_Comm comm, int TAG)
{
  int rank;
  float* temp;
  int* counts = calloc(nslaves+1, sizeof(int));
  int* displacements = calloc(nslaves + 1, sizeof(int));
  if(counts==NULL || displacements==NULL)
    fprintf(stdout,"Unable to allocate memory!");
  for(rank=1; rank <= nslaves; rank++)
    {
      counts[rank] = ldA;
      displacements[rank] = ldA*(rank-1);
    }


  for(rank=1; rank <= nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, TAG, comm);
    }

  MPI_Bcast(xvalue, lenx, MPI_FLOAT, 0, comm);

  MPI_Gatherv(temp, 0, MPI_FLOAT, result, counts, displacements, MPI_FLOAT, 0, comm);

  free(counts); free(displacements);

  return;
}

extern void multiply_ATx(float* xvalue, int lenx, int slave_ldA, int rdA, float* result, int nslaves, MPI_Comm comm, int TAG)
{
  int rank;
  float* temp = calloc(rdA, sizeof(float));
  int* counts = calloc(nslaves+1, sizeof(int));
  int* displacements = calloc(nslaves + 1, sizeof(int));
  if(counts==NULL || displacements==NULL)
    fprintf(stdout,"Unable to allocate memory!");
  for(rank=1; rank <= nslaves; rank++)
    {
      counts[rank] = slave_ldA;
      displacements[rank] = slave_ldA*(rank-1);
    }


  for(rank=1; rank <= nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, TAG, comm);
    }

  //Split xvalue into subvectors and distribute among the slaves
  MPI_Scatterv(xvalue, counts, displacements, MPI_FLOAT, temp, 0, MPI_FLOAT, 0, comm);

  //Sum up the results of each slave multiplication
  MPI_Reduce(temp, result, rdA, MPI_FLOAT, MPI_SUM, 0, comm);

  free(temp); free(counts); free(displacements);

  return;
}

extern void multiply_ATAx(float* xvalue, int lenx, float* result, int nslaves, MPI_Comm comm, int TAG)
{
  int rank;
  float* temp = calloc(lenx, sizeof(float));
  if(temp==NULL)
    fprintf(stdout,"Unable to allocate memory!");


  //Tell slaves we are going to do A^t*A*xvalue
  for(rank=1; rank <= nslaves; rank++)
    {
      MPI_Send(0, 0, MPI_INT, rank, TAG, comm);
    }

  MPI_Bcast(xvalue, lenx, MPI_FLOAT, 0, comm);

  MPI_Reduce(temp, result, lenx, MPI_FLOAT, MPI_SUM, 0, comm);

  free(temp);

  return;
}

extern void get_dat_matrix(float* A, int ldA, int rdA, int myrank, char* filename)
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

  //READ IN MATRIX DATA
  float value;
  int row, col;

  fprintf(stdout, "\nProcess %d getting matrix entries\n", myrank);
  for(row=0; row<ldA; row++) {
    for(col=0; col<rdA; col++) {
      fscanf(matrixfile, "%32f", &value);
      A[row*rdA + col] = value;
      fscanf(matrixfile, " , ");
    }
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
