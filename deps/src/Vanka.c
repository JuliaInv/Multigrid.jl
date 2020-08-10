#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <complex.h>
#include <stdbool.h>

// gcc -O3 -fopenmp -shared -fpic -DBUILD_DLL Vanka.c -o Vanka.dll


#define complex _Complex

// All sorts of options so that conj will just work with the types.
//float  			conj(float x){ return x;}
//double 			conj(double x){ return x;}
//float complex 	conj(float complex x){ return conjf(x);}

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

void getVankaVariablesOfCell(long long *i,long long *n,long long *nf,long long *Idxs,long long includePressure, long long dim){
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



#define ValName FP64
#define IndName INT64
#define spIndType long long
#define spValType double 
#define vankaPrecType float 
#include "Vanka.h"
#undef spIndType
#undef spValType 
#undef vankaPrecType
#undef ValName 
#undef IndName 

#define ValName CFP32
#define IndName INT64
#define spIndType long long
#define spValType float complex 
#define vankaPrecType float complex 
#include "Vanka.h"
#undef spIndType
#undef spValType 
#undef vankaPrecType
#undef ValName 
#undef IndName 


#define ValName CFP64
#define IndName INT64
#define spIndType long long
#define spValType double complex 
#define vankaPrecType float complex 
#include "Vanka.h"
#undef spIndType
#undef spValType 
#undef vankaPrecType
#undef ValName 
#undef IndName  