typedef struct ISTAinstance {
  // PARAMETERS
  float* A;
  int ldA; //left dimension of A
  int rdA; //right dimension of A
  float* b;
  float lambda; //how much weight we give to the 1-norm
  float gamma; //fraction we decrease stepsize each time it fails
  int acceleration; //0 if normal ISTA update; 1 for FISTA acceleration
  char regressionType; // 'l' for linear regression and 'o' for logistic regression

  // VALUES DURING CALCULATION
  float* stepsize;
  float* xcurrent;
  float* xprevious;
  float* searchPoint; //necessary when using FISTA acceleration
  float* gradvalue;
  float* eta; //intermediate space used during calculations, usually holds A*xcurrent

} ISTAinstance;

extern void ISTAsolve(float* A, int ldA, int rdA, float* b, float lambda, float gamma, 
		      int acceleration, char regressionType, float* xvalue, 
		      int MAX_ITER, float MIN_XDIFF, float MIN_FUNCDIFF);

extern void ISTAbacktrack(ISTAinstance* instance);

extern void ISTAgrad(ISTAinstance* instance);

extern float ISTAregress_func(float* xvalue, ISTAinstance* instance);

extern void soft_threshold(float* xvalue, int xlength, float threshold);
