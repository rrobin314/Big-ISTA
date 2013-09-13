extern void ISTAbacktrack(float* A, int ldA, int rdA, float*b, float* searchPoint, float* xcurrent, 
	         	  float* stepsize, float lambda, float gamma, char regressionType);

extern float* ISTAgrad(float* xvalue, float* A, int ldA, int rdA, 
			float* b, char regressionType);

extern float ISTAregress_func(float* xvalue, float* A, int ldA, int rdA, 
			      float* b, char regressionType);

extern void soft_threshold(float* xvalue, int xlength, float threshold);
