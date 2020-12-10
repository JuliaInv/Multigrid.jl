

#define TOKENPASTE(x, y, z) x ## y ##_## z

#define applyHybridKaczmarz(T,S) TOKENPASTE(applyHybridKaczmarz_, T, S)

void applyHybridKaczmarz(ValName,IndName)(spIndType *rowptr , spValType *valA ,spIndType *colA, long long numDomains, long long domainLength ,
						unsigned int *ArrIdxs, spValType *x, spValType *b, long long nrhs, long long n, spValType *invD,long long numit, long long numCores){
	omp_set_num_threads(numCores);
#pragma omp parallel shared(x,b,rowptr,valA,colA,invD,numit,ArrIdxs)
{
	int domain;
	unsigned int row,k,k_rhs,offset;
	spIndType i,gIdx;
	spValType inner;
	for (k=0 ; k < numit ; ++k){
		#pragma omp for
		for (domain=0 ; domain < numDomains ; ++domain){
			for (i=0 ; i < domainLength ; ++i){
				row = ArrIdxs[domain*domainLength + i];
				if (row > 0){
					offset = 0;
					for (k_rhs = 0 ; k_rhs < nrhs ; ++k_rhs){
						inner = b[offset + row-1];
						for (gIdx = rowptr[row-1]-1; gIdx < rowptr[row]-1; ++gIdx){
							// gIdx is in C indices here...
							inner -= conj(valA[gIdx])*x[offset + colA[gIdx]-1];
						}
						inner*=invD[row-1];
						for (gIdx = rowptr[row-1]-1; gIdx < rowptr[row]-1; ++gIdx){
							// gIdx is in C indices here...
							x[offset + colA[gIdx]-1] += inner*valA[gIdx];
						}
						offset += n;
					}
					
				}
			}
		}
	}
}
	return;
}



// void applyHybridGaussSeidel(ValName,IndName)(spIndType *rowptr , spValType *valA ,spIndType *colA, long long numDomains, long long domainLength ,
						// unsigned int *ArrIdxs, spValType *x, spValType *b, spValType *invD,long long numit, long long numCores){
	// omp_set_num_threads(numCores);
// #pragma omp parallel shared(x,b,rowptr,valA,colA,invD,numit,ArrIdxs)
// {
	// int domain;
	// unsigned int row;
	// spIndType k,i,gIdx;
	// spValType inner;
	// for (k=0 ; k < numit ; ++k){
		// #pragma omp for
		// for (domain=0 ; domain < numDomains ; ++domain){
			// for (i=0 ; i < domainLength ; ++i){
				// row = ArrIdxs[domain*domainLength + i];
				// if (row > 0){
					// inner = b[row-1];
					// for (gIdx = rowptr[row-1]-1; gIdx < rowptr[row]-1; ++gIdx){
						// // gIdx is in C indices here...
						// inner -= conj(valA[gIdx])*x[colA[gIdx]-1];
					// }
					// inner*=invD[row-1];
					// x[row-1]+=inner;
				// }
			// }
		// }
	// }
// }
	// return;
// }





