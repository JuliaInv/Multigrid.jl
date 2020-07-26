

#define TOKENPASTE(x, y, z) x ## y ##_## z

#define TOKENPASTE1(x, y) x ## y 

#define computeResidualAtIdx(T,S) TOKENPASTE(computeResidualAtIdx_, T, S)
void computeResidualAtIdx(ValName,IndName)(spIndType *rowptr , spValType *valA ,spIndType *colA, spValType *b, spValType *x,long long* Idxs, spValType *local_r, spIndType blockSize){
	spIndType i,tt;
	for (i = 0 ; i < blockSize ; ++i){
		local_r[i] = b[Idxs[i]-1];
	}
	for (i = 0 ; i < blockSize ; ++i){
		for (tt = rowptr[Idxs[i]-1]; tt <= rowptr[Idxs[i]]-1 ; ++tt){
			local_r[i] -= conj(valA[tt-1])*x[colA[tt-1]-1];
		}
	}	
}

#define computeATvAndAddAtIdx(T,S) TOKENPASTE(computeATvAndAddAtIdx_, T, S)
void computeATvAndAddAtIdx(ValName,IndName)(spIndType *rowptr , spValType *valA ,spIndType *colA, spValType *local_v, spValType *x, long long* Idxs, spIndType blockSize){
	spIndType i,tt;
	for (i = 0 ; i < blockSize ; ++i){
		for (tt = rowptr[Idxs[i]-1]; tt <= rowptr[Idxs[i]]-1 ; ++tt){
			x[colA[tt-1]-1] += valA[tt-1]*local_v[i];
		}
	}	
}




#define updateSolution(T) TOKENPASTE1(updateSolution_, T)
void updateSolution(ValName)(vankaPrecType *mat, spValType *x, spValType *r, int n,long long* Idxs){
	int i,j;
	spValType t;
	for (i=0 ; i<n; ++i){
		t = 0.0;
		for (j=0 ; j<n; ++j){
			t += conj(mat[i*n + j])*r[j];
		}
		x[Idxs[i]-1] += t;
	}
}


#define updateSolutionLocal(T) TOKENPASTE1(updateSolutionLocal_, T)
void updateSolutionLocal(ValName)(vankaPrecType *mat, spValType *x, spValType *r, int n){
	int i,j;
	spValType t;
	for (i=0 ; i<n; ++i){
		t = 0.0;
		for (j=0 ; j<n; ++j){
			t += conj(mat[i*n + j])*r[j];
		}
		x[i] = t;
	}
}




#define RelaxVankaFacesColor(T,S) TOKENPASTE(RelaxVankaFacesColor_, T, S)
void RelaxVankaFacesColor(ValName,IndName)(spIndType *rowptr , spValType *valA ,spIndType *colA,long long *n,long long *nf,long long dim,
							spValType *x, spValType *b, spValType *y, vankaPrecType *D,long long numit,long long includePressure,
							long long lengthVecs, long long numCores){
	int numColors;
	spIndType N,blockSize;
	if (dim==2){
		blockSize = includePressure ? 5 : 4;
		numColors = 4;
		N = n[0]*n[1];
	}else{
		blockSize = includePressure ? 7 : 6;
		numColors = 8;
		N = n[0]*n[1]*n[2];
	}
	omp_set_num_threads(numCores);
#pragma omp parallel shared(x,y,rowptr,valA,colA,n,nf,dim,b,D,numit)
{
	int color;
	spIndType k,i;
	long long *Idxs = (long long*)malloc(blockSize*sizeof(long long));
	long long *i_vec = (long long*)malloc(dim*sizeof(long long));
	spValType* local_r = (spValType*)malloc(blockSize*sizeof(spValType));
	for (k=0 ; k < numit ; ++k){
		//for (color=1 ; color <= numColors ; ++color){
		for (color=1 ; color <= 2 ; ++color){
			//#pragma omp single // if we wish to make the code identical for parallel and serial, the residual has to be computed with a copy of x.
			//{
			//	memcpy(y,x,lengthVecs*sizeof(spValType));
			//}
			#pragma omp for
			for (i=1 ; i <= N ; ++i){
				cs2loc(i,n,dim,i_vec);
				if (cellColor(i_vec,dim)==color){
					getVankaVariablesOfCell(i_vec,n,nf,Idxs,includePressure,dim);
					computeResidualAtIdx(ValName,IndName)(rowptr,valA,colA,b,x,Idxs,local_r,blockSize);
					updateSolution(ValName)(D + (i-1)*blockSize*blockSize , x, local_r, blockSize,Idxs);
				}
			}
		}
	}
	free(Idxs);
	free(i_vec);
	free(local_r);
}
	return;
}

// double computeNorm(spValType* x, int len){
	// double norm = 0.0;
	// for (int i = 0 ; i < len ; ++i){
		// norm += creal(conj(x[i])*x[i]);
		// norm += creal(x[i])*creal(x[i]) + cimag(x[i])*cimag(x[i]);
	// }
	// return norm;
// }


#define applyHybridCellWiseKaczmarz(T,S) TOKENPASTE(applyHybridCellWiseKaczmarz_, T, S)
void applyHybridCellWiseKaczmarz(ValName,IndName)(spIndType *rowptr , spValType *valA ,spIndType *colA,long long *n,long long *nf,long long dim,
								spValType *x, spValType *b, vankaPrecType *D,long long numit,long long includePressure,
								long long numCores, unsigned int *ArrIdxs,long long numDomains, long long domainLength){
	spIndType N,blockSize;
	if (dim==2){
		blockSize = includePressure ? 5 : 4;
		N = n[0]*n[1];
	}else{
		blockSize = includePressure ? 7 : 6;
		N = n[0]*n[1]*n[2];
	}
	omp_set_num_threads(numCores);
#pragma omp parallel shared(x,rowptr,valA,colA,n,nf,dim,b,D,numit)
{
	spIndType k,i;
	long long cell,domain;
	long long *Idxs = (long long*)malloc(blockSize*sizeof(long long));
	long long *cell_vec = (long long*)malloc(dim*sizeof(long long));
	spValType* local_r = (spValType*)malloc(blockSize*sizeof(spValType));
	spValType* local_x = (spValType*)malloc(blockSize*sizeof(spValType));
	for (k=0 ; k < numit ; ++k){
		#pragma omp for
		for (domain=0 ; domain < numDomains ; ++domain){
			for (i=0 ; i < domainLength ; ++i){
				cell = (long long)ArrIdxs[domain*domainLength + i];
				if (cell > 0){
					cs2loc(cell,n,dim,cell_vec);
					getVankaVariablesOfCell(cell_vec,n,nf,Idxs,includePressure,dim);
					// printf("Cell #%ld::",cell);
					computeResidualAtIdx(ValName,IndName)(rowptr,valA,colA,b,x,Idxs,local_r,blockSize);
					// printf("rnorm %6.3e, ",computeNorm(local_r,blockSize));
					updateSolutionLocal(ValName)(D + (cell-1)*blockSize*blockSize , local_x, local_r, blockSize);
					// printf("xnorm %6.3e, ",computeNorm(local_x,blockSize));
					// printf("Dnorm %6.3e",computeNorm(D,blockSize*blockSize));
					computeATvAndAddAtIdx(ValName,IndName)(rowptr,valA,colA,local_x,x,Idxs,blockSize);
					// printf("\n");
				}
			}
			// if (includePressure){
				// double complex inner;
				// double complex invD;
				// spIndType row;
				// spIndType gIdx;
				// for (i=0 ; i < domainLength ; ++i){
					// cell = (long long)ArrIdxs[domain*domainLength + i];
					// if (cell > 0){
						// invD = 0.0;
						// cs2loc(cell,n,dim,cell_vec);
						// getVankaVariablesOfCell(cell_vec,n,nf,Idxs,includePressure,dim);
						// row = Idxs[blockSize-1];
						// inner = b[row-1];
						// for (gIdx = rowptr[row-1]-1; gIdx < rowptr[row]-1; ++gIdx){
							// // gIdx is in C indices here...
							// inner -= conj(valA[gIdx])*x[colA[gIdx]-1];
							// invD += creal(conj(valA[gIdx])*valA[gIdx]);
						// }
						// inner*=(1.0/invD);
						// for (gIdx = rowptr[row-1]-1; gIdx < rowptr[row]-1; ++gIdx){
							// // gIdx is in C indices here...
							// x[colA[gIdx]-1] += inner*valA[gIdx];
						// }
					// }
				
				// }
			// }  
		}
	}
	free(Idxs);
	free(cell_vec);
	free(local_r);
	free(local_x);
}
	return;
}
