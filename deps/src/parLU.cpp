#include <iostream>
#include <omp.h>
#include <complex.h>

#define complex _Complex


// All sorts of options so that conj will just work with the types.
float  			conj(float x){ return x;}
double 			conj(double x){ return x;}
float complex 	conj(float complex x){ return conjf(x);}
 

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif



template<typename IND, typename VAL>
void applyLUsolve(long long *rowptrL , VAL *valL ,IND *colL,long long *rowptrU , VAL *valU, 
					IND *colU, IND *p, IND *q,  long long *n, long long *nnz, 
					VAL *x, VAL *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores);
		
template<typename IND, typename VAL>
void applyLUsolveTrans(long long *rowptrL , VAL *valL ,IND *colL,long long *rowptrU , VAL *valU, 
					IND *colU, IND *p, IND *q,  long long *n, long long *nnz, 
					VAL *x, VAL *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores);


#define IND_T unsigned int
#define VAL_T float
EXTERNC void applyLUsolve_FP32_UINT32(long long *rowptrL , VAL_T *valL ,IND_T *colL,long long *rowptrU , VAL_T *valU ,IND_T *colU,
							IND_T *p, IND_T *q,  long long *n, long long *nnz, 
						    VAL_T *x, VAL_T *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores, long long doTranspose);
void applyLUsolve_FP32_UINT32(long long *rowptrL , VAL_T *valL ,IND_T *colL, long long *rowptrU , VAL_T *valU ,IND_T *colU,
							IND_T *p, IND_T *q,  long long *n, long long *nnz, 
						    VAL_T *x, VAL_T *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores, long long doTranspose){
	if (doTranspose==0){
		applyLUsolve<IND_T,VAL_T>(rowptrL , valL ,colL,rowptrU , valU ,colU,p,q,n,nnz,x,b,num_LUs,num_rhs,numCores,jointCores);
	}else{
		applyLUsolveTrans<IND_T,VAL_T>(rowptrL , valL ,colL,rowptrU , valU ,colU,p,q,n,nnz,x,b,num_LUs,num_rhs,numCores,jointCores);
	}
}
#undef IND_T
#undef VAL_T

#define IND_T long long
#define VAL_T double 
EXTERNC void applyLUsolve_FP64_INT64(long long *rowptrL , VAL_T *valL ,IND_T *colL,long long *rowptrU , VAL_T *valU ,IND_T *colU,
							IND_T *p, IND_T *q,  long long *n, long long *nnz, 
						    VAL_T *x, VAL_T *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores, long long doTranspose);
void applyLUsolve_FP64_INT64(long long *rowptrL , VAL_T *valL ,IND_T *colL, long long *rowptrU , VAL_T *valU ,IND_T *colU,
							IND_T *p, IND_T *q,  long long *n, long long *nnz, 
						    VAL_T *x, VAL_T *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores, long long doTranspose){
	if (doTranspose==0){
		applyLUsolve<IND_T,VAL_T>(rowptrL , valL ,colL,rowptrU , valU ,colU,p,q,n,nnz,x,b,num_LUs,num_rhs,numCores,jointCores);
	}else{
		applyLUsolveTrans<IND_T,VAL_T>(rowptrL , valL ,colL,rowptrU , valU ,colU,p,q,n,nnz,x,b,num_LUs,num_rhs,numCores,jointCores);
	}
}
#undef IND_T
#undef VAL_T

#define IND_T long long
#define VAL_T double complex
EXTERNC void applyLUsolve_CFP64_INT64(long long *rowptrL, VAL_T *valL, IND_T *colL,long long *rowptrU , VAL_T *valU ,IND_T *colU,
							IND_T *p, IND_T *q,  long long *n, long long *nnz, 
						    VAL_T *x, VAL_T *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores, long long doTranspose);
void applyLUsolve_CFP64_INT64(long long *rowptrL , VAL_T *valL ,IND_T *colL, long long *rowptrU , VAL_T *valU ,IND_T *colU,
							IND_T *p, IND_T *q,  long long *n, long long *nnz, 
						    VAL_T *x, VAL_T *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores, long long doTranspose){
	if (doTranspose==0){
		applyLUsolve<IND_T,VAL_T>(rowptrL , valL ,colL,rowptrU , valU ,colU,p,q,n,nnz,x,b,num_LUs,num_rhs,numCores,jointCores);
	}else{
		applyLUsolveTrans<IND_T,VAL_T>(rowptrL , valL ,colL,rowptrU , valU ,colU,p,q,n,nnz,x,b,num_LUs,num_rhs,numCores,jointCores);
	}
}
#undef IND_T
#undef VAL_T


#define IND_T unsigned int
#define VAL_T float complex 
EXTERNC void applyLUsolve_CFP32_UINT32(long long *rowptrL , VAL_T *valL ,IND_T *colL,long long *rowptrU , VAL_T *valU ,IND_T *colU,
							IND_T *p, IND_T *q,  long long *n, long long *nnz, 
						    VAL_T *x, VAL_T *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores, long long doTranspose);
void applyLUsolve_CFP32_UINT32(long long *rowptrL , VAL_T *valL ,IND_T *colL, long long *rowptrU , VAL_T *valU ,IND_T *colU,
							IND_T *p, IND_T *q,  long long *n, long long *nnz, 
						    VAL_T *x, VAL_T *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores, long long doTranspose){
	if (doTranspose==0){
		applyLUsolve<IND_T,VAL_T>(rowptrL , valL ,colL,rowptrU , valU ,colU,p,q,n,nnz,x,b,num_LUs,num_rhs,numCores,jointCores);
	}else{
		applyLUsolveTrans<IND_T,VAL_T>(rowptrL , valL ,colL,rowptrU , valU ,colU,p,q,n,nnz,x,b,num_LUs,num_rhs,numCores,jointCores);
	}
}
#undef IND_T
#undef VAL_T

#define IND_T long long
#define VAL_T float complex 
EXTERNC void applyLUsolve_CFP32_INT64(long long *rowptrL , VAL_T *valL ,IND_T *colL,long long *rowptrU , VAL_T *valU ,IND_T *colU,
							IND_T *p, IND_T *q,  long long *n, long long *nnz, 
						    VAL_T *x, VAL_T *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores, long long doTranspose);
void applyLUsolve_CFP32_INT64(long long *rowptrL , VAL_T *valL ,IND_T *colL, long long *rowptrU , VAL_T *valU ,IND_T *colU,
							IND_T *p, IND_T *q,  long long *n, long long *nnz, 
						    VAL_T *x, VAL_T *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores, long long doTranspose){
	if (doTranspose==0){
		applyLUsolve<IND_T,VAL_T>(rowptrL , valL ,colL,rowptrU , valU ,colU,p,q,n,nnz,x,b,num_LUs,num_rhs,numCores,jointCores);
	}else{
		applyLUsolveTrans<IND_T,VAL_T>(rowptrL , valL ,colL,rowptrU , valU ,colU,p,q,n,nnz,x,b,num_LUs,num_rhs,numCores,jointCores);
	}
}
#undef IND_T
#undef VAL_T


// n[i] the number of 

template<typename IND, typename VAL>
void applyLUsolve(long long *rowptrL , VAL *valL ,IND *colL,long long *rowptrU , VAL *valU ,IND *colU, 
					IND *p, IND *q,  long long *n, long long *nnz, 
					VAL *x, VAL *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores){
omp_set_num_threads(numCores);
#pragma omp parallel shared(x,b,rowptrL,rowptrU,valL,valU,colL,colU,p,q,n,nnz)
{
	long long sysIdx;
	#pragma omp for
	for (sysIdx=0 ; sysIdx < num_LUs*num_rhs ; ++sysIdx){
		// Here we assume that the RHS's are consecutive by LUs. Each num_rhs we skip an LU
		long long rhsIdx = sysIdx%num_rhs;
		long long LUIdx  = sysIdx/num_rhs;
		long long LUoffset = 0;
		long long RHSoffset = 0;
		long long PTRoffset = 0;
		long long PERMoffset = 0;
		for (long long i=0; i < LUIdx; ++i){
			LUoffset +=2*nnz[i];
			PTRoffset+=2*(n[i]+1);
			PERMoffset+=n[i];
			RHSoffset+=num_rhs*n[i];
		}
		for (long long i=0; i < rhsIdx; ++i){
			RHSoffset+=n[rhsIdx];
		}
		long long *rowptr   = rowptrL + LUoffset;
		VAL *val            = valL + LUoffset;
		IND *col            = colL + LUoffset;
		IND *pi  		    = p + PERMoffset;
		IND *qi  		    = q + PERMoffset;
		VAL *bi 			= b + RHSoffset;
		VAL *xi 			= x + RHSoffset;
		IND gIdx 		= 0;
		VAL diagElem;
		VAL inner_prod;
		// FORWARD SUBSTITUTION:
		
		for (long long row=0 ; row < (n[LUIdx]) ; ++row){
			// when transposed, and hence held in CSR, the diagonal elements of L are last.
			inner_prod = (bi[pi[row]-1]);
			for (gIdx = rowptr[row]-1; gIdx < rowptr[row+1]-2; ++gIdx){ // only until the second to last
				// gIdx is in C indices here...
				inner_prod -= (val[gIdx])*xi[col[gIdx]-1];
			}
			diagElem = val[rowptr[row+1]-2]; // "-2" once for julia 1-based indexing, once for last element
			xi[row] = inner_prod/diagElem;
		}
		rowptr = rowptrU + LUoffset;
		val    = valU + LUoffset;
		col    = colU + LUoffset;
		// BACKWARD SUBSTITUTION:
		for (long long row = (n[LUIdx]-1) ; row >= 0 ; --row){
			// when transposed, and hence held in CSR, the diagonal elements of U are first.
			diagElem = val[rowptr[row]-1]; // "-1" for julia 1-based indexing
			inner_prod = xi[row];
			for (gIdx = rowptr[row]; gIdx < rowptr[row+1]-1; ++gIdx){ // starting from the second element
				// gIdx is in C indices here...
				inner_prod -= (val[gIdx])*bi[col[gIdx]-1];
			}
			bi[row] = inner_prod/diagElem;
		} 
		for (long long row = 0 ; row < n[LUIdx] ; ++row){
			xi[qi[row]-1] = bi[row];
		}
	}
}
	return;
}


template<typename IND, typename VAL>
void applyLUsolveTrans(long long *rowptrL , VAL *valL ,IND *colL,long long *rowptrU , VAL *valU ,IND *colU, 
					IND *p, IND *q,  long long *n, long long *nnz, 
					VAL *x, VAL *b, long long num_LUs, long long num_rhs, long long numCores, long long jointCores){
omp_set_num_threads(numCores);
#pragma omp parallel shared(x,b,rowptrL,rowptrU,valL,valU,colL,colU,p,q,n,nnz)
{
	long long sysIdx;
	#pragma omp for
	for (sysIdx=0 ; sysIdx < num_LUs*num_rhs ; ++sysIdx){
		// Here we assume that the RHS's are consecutive by LUs. Each num_rhs we skip an LU
		long long rhsIdx = sysIdx%num_rhs;
		long long LUIdx  = sysIdx/num_rhs;
		long long LUoffset = 0;
		long long RHSoffset = 0;
		long long PTRoffset = 0;
		long long PERMoffset = 0;
		for (long long i=0; i < LUIdx; ++i){
			LUoffset +=2*nnz[i];
			PTRoffset+=2*(n[i]+1);
			PERMoffset+=n[i];
			RHSoffset+=num_rhs*n[i];
		}
		for (long long i=0; i < rhsIdx; ++i){
			RHSoffset+=n[rhsIdx];
		}
		long long *rowptr   = rowptrU + LUoffset;
		VAL *val            = valU + LUoffset;
		IND *col            = colU + LUoffset;
		IND *pi  		    = p + PERMoffset; // p and q replace roles in transpose.
		IND *qi  		    = q + PERMoffset;
		VAL *bi 			= b + RHSoffset;
		VAL *xi 			= x + RHSoffset;
		IND gIdx 		= 0;
		VAL diagElem;
		VAL t;
		for (long long row=0 ; row < (n[LUIdx]) ; ++row){
			xi[row] = bi[qi[row]-1];
		}
		// FORWARD SUBSTITUTION CSC: U is held in CSR, hence U^T is CSC.
		for (long long row=0 ; row < (n[LUIdx]) ; ++row){
			// when held in CSR, the diagonal elements of U are first.
			diagElem = conj(val[rowptr[row]-1]); // "-1" for julia 1-based indexing
			bi[row] = xi[row] / diagElem;
			t = bi[row];
			for (gIdx = rowptr[row]; gIdx < rowptr[row+1]-1; ++gIdx){ // starting from the second element
				// gIdx is in C indices here...
				xi[col[gIdx]-1] -= conj(val[gIdx])*t;
			}
		}
		rowptr = rowptrL + LUoffset;
		val    = valL + LUoffset;
		col    = colL + LUoffset;
		// BACKWARD SUBSTITUTION:
		for (long long row = (n[LUIdx]-1) ; row >= 0 ; --row){
			// when held in CSR, the diagonal elements of L are last.
			diagElem = conj(val[rowptr[row+1]-2]); // "-2" once for julia 1-based indexing, once for last element
			t = (bi[row]) / diagElem;
			xi[pi[row]-1] = t;
			for (gIdx = rowptr[row]-1; gIdx < rowptr[row+1]-2; ++gIdx){ // only until the second to last
				// gIdx is in C indices here...
				bi[col[gIdx]-1] -= conj(val[gIdx])*t;
			}
		} 
	}
}
	return;
}



