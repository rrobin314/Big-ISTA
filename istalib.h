typedef struct ISTAinstance {
  // PARAMETERS
  float* A;
  int ldA; //left dimension of A
  int rdA; //right dimension of A
  float* b;
  float lambda; //how much weight we give to the 1-norm
  float gamma; //fraction we decrease stepsize each time it fails
  int acceleration; //0 if normal ISTA update; 1 for FISTA acceleration
  int interceptFlag; //1 to add a column of ones to A to calculate an intercept
  char regressionType; // 'l' for linear regression and 'o' for logistic regression

  // SCALING VALUES
  float* meanShifts;
  float* scalingFactors;
  float intercept;

  // VALUES DURING CALCULATION
  float* stepsize;
  float* xcurrent;
  float* xprevious;
  float* searchPoint; //necessary when using FISTA acceleration
  float* gradvalue;
  float* eta; //intermediate space used during calculations, usually holds A*xcurrent

} ISTAinstance;

// "Constructor" for ISTAinstance
// Returns a pointer to an ISTAinstance object with the arguments to the functions set as the corresponding
// elements in the ISTAinstance object.  Also, allocates appropriate memory for stepsize, xprevious,
// searchPoint, gradvalue, and eta.  Finally, sets searchPoint equal to xvalue.
extern ISTAinstance* ISTAinstance_new(float* A, int ldA, int rdA, float* b, float lambda, float gamma, 
				      int acceleration, int interceptFlag, char regressionType, float* xvalue, float step );

// "Deconstructor" for ISTAinstance
// Applies free to all pointers contained in instance, then applies free to instance itself.
extern void ISTAinstance_free(ISTAinstance* instance);

// Rescales the columns of A to have zero mean and unit norm.
// Stores the shifts and rescaling factors in the variables
// "meanShifts" and "scalingFactors" respectively.
extern void ISTArescale(ISTAinstance* instance);

// Converts the scaled problem back into its original,
// unscaled form.  This automatically implies the use of 
// an intercept, whose value is stored in "intercept"
extern void ISTAundoRescale(ISTAinstance* instance);

extern void ISTAaddIntercept(ISTAinstance* instance);


// This version of ISTAsolve does not allocate any memory
// and is meant to be used with ISTAinstance_new and ISTAinstance_free to handle
// memory allocation.
extern void ISTAsolve_lite(ISTAinstance* instance, int MAX_ITER, float MIN_FUNCDIFF );


// Backtracking routine to determine how big of a gradient step to take during ISTA.
// Does the following updates:
// Updates gradvalue to that of current searchPoint
// Updates xcurrent to the gradient step from searchPoint indicated by stepsize
// Updates eta to xcurrent - searchPoint
// If additional loops are necessary, updates stepsize to gamma*stepsize 
extern void ISTAbacktrack(ISTAinstance* instance);


// Calculates gradient of ISTAregress_func at searchPoint and stores it in gradvalue
extern void ISTAgrad(ISTAinstance* instance);


// Calculates the appropriate regression function for either linear or logistic regression
extern float ISTAregress_func(float* xvalue, ISTAinstance* instance);

extern void soft_threshold(float* xvalue, int xlength, float threshold);

// Calculates a path of lambdas to solve on:
// If lambdaStart > 0, then our starting path is just lambdaStart
// If lambdaStart < 0, then we calculate our starting point to be:
//     0.5 * || A' * b ||_infinity
// 
// If numLambdas == 1, then our path just consists of lambdaFinish
// If numLambdas > 1, then we draw an exponential curve between
// the starting point and lambdaFinish
extern void calcLambdas(float* lambdas, int numLambdas, float lambdaStart, 
			float lambdaFinish, float* A, int ldA, int rdA, 
			float* b, float* result);

