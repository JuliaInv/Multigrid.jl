#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <complex.h>
#include <stdbool.h>

#define spIndType long long
#define spValType float complex 
#define vankaPrecType float complex 

// gcc -O3 -fopenmp -shared -fpic -DBUILD_DLL Vanka.c -o Vanka.dll

void cs2loc(long long cs_loc,long long *n,long long dim, long long *loc){
	if (dim==3){
		loc[0] = (cs_loc-1)%n[0] + 1;
		loc[1] = (((cs_loc-1)%(n[0]*n[1]))/n[0]) + 1;
		loc[2] = (cs_loc-1)/(n[0]*n[1]) + 1;
	}else{
		loc[0] = (cs_loc-1)%n[0] + 1;
		loc[1] = (cs_loc-1)/n[0] + 1;
	}
	return;
}

int cellColor(long long *i,long long dim){
	if (dim==2){
		/// 4-color
		//if (i[0]%2 == 1){ 
		//	 return i[1]%2==1 ? 1 : 2; 
		// }else{ 
		//	 return i[1]%2==1 ? 3 : 4; 
		// }
		/// Red-Black
		if (i[0]%2 == 1){ 
			return i[1]%2==1 ? 1 : 2; 
		}else{ 
			return i[1]%2==1 ? 2 : 1; 
		}
		
		
	}else{
		/// 8-color
		// if (i[0]%2 == 1){ 
			// if (i[1]%2==1){
				// return i[2]%2==1 ? 1 : 2; 
			// }else{
				// return i[2]%2==1 ? 3 : 4;
			// }
		// }else{ 
			// if (i[1]%2==1){
				// return i[2]%2==1 ? 5 : 6; 
			// }else{
				// return i[2]%2==1 ? 7 : 8;
			// }
		// }
		
		/// Red-black
		if (i[0]%2 == 1){ 
			if (i[1]%2==1){
				return i[2]%2==1 ? 1 : 2; 
			}else{
				return i[2]%2==1 ? 2 : 1;
			}
		}else{ 
			if (i[1]%2==1){
				return i[2]%2==1 ? 2 : 1; 
			}else{
				return i[2]%2==1 ? 1 : 2;
			}
		}
		
		
	}		
}

long long loc2cs3D(long long *loc,long long n1,long long n2,long long n3){
return loc[0] + (loc[1]-1)*n1 + (loc[2]-1)*n1*n2;
}

void getVankaVariablesOfCell(long long *i,long long *n,long long *nf,spIndType *Idxs,long long includePressure, long long dim){
long long t1,t2,t3;
if (includePressure){
	if (dim==2){	
		t1 = i[0] + (i[1]-1)*(n[0]+1);
		t2 = nf[0] + i[0] + (i[1]-1)*n[0];
		Idxs[0] = t1;
		Idxs[1] = t1+1;
		Idxs[2] = t2;
		Idxs[3] = t2 + n[0];
		Idxs[4] = nf[1] + t2;
	}else{
		t1 = loc2cs3D(i,n[0]+1,n[1],n[2]); 
		Idxs[0] = t1;
		Idxs[1] = t1+1;
		t2 = nf[0] + loc2cs3D(i,n[0],n[1]+1,n[2]);
		Idxs[2] = t2;
		Idxs[3] = t2+n[0];
		t3 = nf[0] + nf[1] + loc2cs3D(i,n[0],n[1],n[2]+1);
		Idxs[4] = t3;
		Idxs[5] = t3+n[0]*n[1];
		Idxs[6] = nf[2] + t3;
	}
}else{
	if (dim==2){	
		t1 = i[0] + (i[1]-1)*(n[0]+1);
		t2 = nf[0] + i[0] + (i[1]-1)*n[0];
		Idxs[0] = t1;
		Idxs[1] = t1+1;
		Idxs[2] = t2;
		Idxs[3] = t2 + n[0];
	}else{
		t1 = loc2cs3D(i,n[0]+1,n[1],n[2]); 
		Idxs[0] = t1;
		Idxs[1] = t1+1;
		t2 = nf[0] + loc2cs3D(i,n[0],n[1]+1,n[2]);
		Idxs[2] = t2;
		Idxs[3] = t2+n[0];
		t3 = nf[0] + nf[1] + loc2cs3D(i,n[0],n[1],n[2]+1);
		Idxs[4] = t3;
		Idxs[5] = t3+n[0]*n[1];
	}
}
return;
}

void computeResidualAtIdx(spIndType *rowptr , spValType *valA ,spIndType *colA,spValType *b,spValType *x,spIndType* Idxs, spValType *local_r, spIndType blockSize){
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


void computeATvAndAddAtIdx(spIndType *rowptr , spValType *valA ,spIndType *colA,spValType *local_v,spValType *x,spIndType* Idxs, spIndType blockSize){
	spIndType i,tt;
	for (i = 0 ; i < blockSize ; ++i){
		for (tt = rowptr[Idxs[i]-1]; tt <= rowptr[Idxs[i]]-1 ; ++tt){
			x[colA[tt-1]-1] += valA[tt-1]*local_v[i];
		}
	}	
}





void updateSolution(vankaPrecType *mat, spValType *x, spValType *r, int n,spIndType* Idxs){
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

void updateSolutionLocal(vankaPrecType *mat, spValType *x, spValType *r, int n){
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


void RelaxVankaFacesColor(spIndType *rowptr , spValType *valA ,spIndType *colA,long long *n,long long *nf,long long dim,
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
	spIndType *Idxs = (spIndType*)malloc(blockSize*sizeof(spIndType));
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
					computeResidualAtIdx(rowptr,valA,colA,b,x,Idxs,local_r,blockSize);
					updateSolution(D + (i-1)*blockSize*blockSize , x, local_r, blockSize,Idxs);
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




void applyHybridCellWiseKaczmarz(spIndType *rowptr , spValType *valA ,spIndType *colA,long long *n,long long *nf,long long dim,
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
	spIndType *Idxs = (spIndType*)malloc(blockSize*sizeof(spIndType));
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
					computeResidualAtIdx(rowptr,valA,colA,b,x,Idxs,local_r,blockSize);
					// printf("rnorm %6.3e, ",computeNorm(local_r,blockSize));
					updateSolutionLocal(D + (cell-1)*blockSize*blockSize , local_x, local_r, blockSize);
					// printf("xnorm %6.3e, ",computeNorm(local_x,blockSize));
					// printf("Dnorm %6.3e",computeNorm(D,blockSize*blockSize));
					computeATvAndAddAtIdx(rowptr,valA,colA,local_x,x,Idxs,blockSize);
					// printf("\n");
				}
			}
			 if (includePressure){
				double complex inner;
				double complex invD;
				spIndType row;
				spIndType gIdx;
				for (i=0 ; i < domainLength ; ++i){
					cell = (long long)ArrIdxs[domain*domainLength + i];
					if (cell > 0){
						invD = 0.0;
						cs2loc(cell,n,dim,cell_vec);
						getVankaVariablesOfCell(cell_vec,n,nf,Idxs,includePressure,dim);
						row = Idxs[blockSize-1];
						inner = b[row-1];
						for (gIdx = rowptr[row-1]-1; gIdx < rowptr[row]-1; ++gIdx){
							// gIdx is in C indices here...
							inner -= conj(valA[gIdx])*x[colA[gIdx]-1];
							invD += creal(conj(valA[gIdx])*valA[gIdx]);
						}
						inner*=(1.0/invD);
						for (gIdx = rowptr[row-1]-1; gIdx < rowptr[row]-1; ++gIdx){
							// gIdx is in C indices here...
							x[colA[gIdx]-1] += inner*valA[gIdx];
						}
					}
				
				}
			}  
			
			
			
			
		}
	}
	free(Idxs);
	free(cell_vec);
	free(local_r);
	free(local_x);
}
	return;
}









int main(int argc, char **argv){	
   return 0;
  }

	

