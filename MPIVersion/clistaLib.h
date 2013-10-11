typedef struct ISTAinstance_mpi {
  // PARAMETERS
  //float* A; A is now stored with the slaves
  int ldA; //left dimension of A
  int rdA; //right dimension of A
  float* b;
  float lambda; //how much weight we give to the 1-norm
  float gamma; //fraction we decrease stepsize each time it fails
  int acceleration; //0 if normal ISTA update; 1 for FISTA acceleration
  char regressionType; // 'l' for linear regression and 'o' for logistic regression

  // MPI VALUES
  int nslaves;
  MPI_Comm comm;
  int tag_ax;
  int tag_atax;
  int tag_atx;
  int tag_die;
  
  // VALUES DURING CALCULATION
  float* stepsize;
  float* xcurrent;
  float* xprevious;
  float* searchPoint; //necessary when using FISTA acceleration
  float* gradvalue;
  float* eta; //intermediate space used during calculations, usually holds A*xcurrent

} ISTAinstance_mpi;

// "Constructor" for ISTAinstance
// Returns a pointer to an ISTAinstance object with the arguments to the functions set as the corresponding
// elements in the ISTAinstance object.  Also, allocates appropriate memory for stepsize, xprevious,
// searchPoint, gradvalue, and eta.  Finally, sets searchPoint equal to xvalue.
extern ISTAinstance_mpi* ISTAinstance_mpi_new(int ldA, int rdA, float* b, float lambda, float gamma, 
				      int acceleration, char regressionType, float* xvalue, float step );

// "Deconstructor" for ISTAinstance
// Applies free to all pointers contained in instance, then applies free to instance itself.
extern void ISTAinstance_mpi_free(ISTAinstance_mpi* instance);


/*


// Applies ISTA to min( ISTAregress_func(xvalue) + lambda*regFunc(xvalue) )
//    where ISTAregress_func is ||Ax-b||^2 for linear regression and the logistic function for logistic regression
//    and regFunc is the L1 norm
// The result is recorded by updating the xvalue pointer.
// The function allocates and deallocates memory for xprevious, searchPoint, gradvalue, and eta
// and, hence, may not be efficient for situations where many calls to this function are necessary
extern void ISTAsolve(float* A, int ldA, int rdA, float* b, float lambda, float gamma, 
		      int acceleration, char regressionType, float* xvalue, 
		      int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF);

// This version of ISTAsolve does not allocate any memory
// and is meant to be used with ISTAinstance_new and ISTAinstance_free to handle
// memory allocation.
extern void ISTAsolve_lite(ISTAinstance* instance, int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF );

// Applies ISTAsolve_lite to instance for a series of lambdas.
// All solutions are returned in the double pointer.
// Memory for this is allocated in the function, but not deallocated - CAREFUL to deallocate memory
extern float** ISTAsolve_pathwise(float* lambdas, int num_lambdas, ISTAinstance* instance, 
				  int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF );

// Cross validation routine.
// folds is an integer array of length ldA with values from 0 to num_folds - 1
// that determines which rows are in which fold.
// Then for each fold, the code runs ISTA on the rows NOT in that fold and 
// gets a solution "x".  Then it calculates the average value of the regression function at "x"
// for the rows in the fold.  This is the error for that fold.
// The final error is the average of the fold errors.
extern float ISTAcrossval(ISTAinstance* instance, int* folds, int num_folds, 
			  int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF );

// Backtracking routine to determine how big of a gradient step to take during ISTA.
// Does the following updates:
// Updates gradvalue to that of current searchPoint
// Updates xcurrent to the gradient step from searchPoint indicated by stepsize
// Updates eta to xcurrent - searchPoint
// If additional loops are necessary, updates stepsize to gamma*stepsize 
extern void ISTAbacktrack(ISTAinstance* instance);

// Version of backtracking for cross validation
extern void ISTAbacktrack_cv(ISTAinstance* instance, int currentFold, int* folds);

// Calculates gradient of ISTAregress_func at searchPoint and stores it in gradvalue
extern void ISTAgrad(ISTAinstance* instance);

// Version of gradient method for cross validation
extern void ISTAgrad_cv(ISTAinstance* instance, int currentFold, int* folds);

*/


// Calculates the appropriate loss function for either linear or logistic regression
extern float ISTAloss_func_mpi(float* xvalue, ISTAinstance_mpi* instance);

// Calculates the regression function value using only the rows corresponding to currentFold in folds
//extern float ISTAregress_func_cv(float* xvalue, ISTAinstance* instance, int currentFold, int* folds, int insideFold);

extern void soft_threshold(float* xvalue, int xlength, float threshold);

extern void multiply_Ax(float* xvalue, int lenx, int ldA, float* result, int nslaves, MPI_Comm comm, int TAG);

extern void multiply_ATAx(float* xvalue, int lenx, float* result, int nslaves, MPI_Comm comm, int TAG);

extern void get_dat_matrix(float* A, int ldA, int rdA, int myrank);
