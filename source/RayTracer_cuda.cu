////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//#include <stdio.h>
#include "RayTracer.h"
//#include "helper_cuda.h"


__global__ void RayTracer(uchar4* dest, const int imageW, const int imageH)
{
	const int ix = threadIdx.x;
	const int iy = threadIdx.y;

	const int pixel = imageW * iy + ix;

	dest[pixel].y = 100;
}


void RunRayTracer(uchar4* dest, const int imageW, const int imageH)
{
	dim3 numThreads(8, 8);

	RayTracer<<<10, numThreads>>>(dest, imageW, imageH);
}