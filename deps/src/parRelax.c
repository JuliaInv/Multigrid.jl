#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <complex.h>



#define ValName CFP64
#define IndName INT64
#define spIndType long long
#define spValType double complex
#include"parRelax.h"
#undef spIndType
#undef spValType
#undef ValName
#undef IndName


#define ValName CFP32
#define IndName INT64
#define spIndType long long
#define spValType float complex
#include"parRelax.h"
#undef spIndType
#undef spValType
#undef ValName
#undef IndName


#define ValName FP64
#define IndName INT64
#define spValType double
#define spIndType long long
#include"parRelax.h"
#undef spIndType
#undef spValType
#undef ValName
#undef IndName


#define ValName FP32
#define IndName INT64
#define spValType float
#define spIndType long long
#include"parRelax.h"
#undef spIndType
#undef spValType
#undef ValName
#undef IndName
