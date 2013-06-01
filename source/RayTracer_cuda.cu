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
#include "helper_math.h"


__global__ void RayTracer(uchar4* dest, const int imageW, const int imageH, float4 cameraLocation, float4 cameraUp, float4 cameraForward, float4 cameraRight, float nearPlaneDistance, float2 viewSize)
{
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;

	// Compute the location in the dest array that will be written to
	const int pixel = imageW * iy + ix;

	// Compute the center of the near plane. All rays will be computed as an offset from this point
	const float4 lookAt = cameraLocation + cameraForward * nearPlaneDistance;

	const float4 rayMidPoint = lookAt + cameraRight * ((float(ix) / float(imageW) - 0.5f) * viewSize.x) + cameraUp * ((1 - (float(iy) / float(imageH)) - 0.5f) * viewSize.y); 
	const float4 ray = normalize(rayMidPoint - cameraLocation);

	const float4 sphereCenter = make_float4(0, 0, 20, 1);
	const float radius = 10.0f;

	const float4 rayOriginMinusSphereCenter = cameraLocation - sphereCenter;

	const float A = dot(ray, ray);
	const float B = 2 * dot(rayOriginMinusSphereCenter, ray);
	const float C = dot(rayOriginMinusSphereCenter, rayOriginMinusSphereCenter) - radius * radius;

	const float disc = B * B - 4 * A * C;

	float t = -1.0f;

	if(disc >= 0)
	{
		const float discSqrt = sqrtf(disc);
		float q;
		
		if(B < 0)
		{
			q = (-B - discSqrt) / 2.0f;
		}
		else
		{
			q = (-B + discSqrt) / 2.0f;
		}

		float t0 = q / A;
		float t1 = C / q;

		if(t0 > t1)
		{
			float temp = t0;
			t0 = t1;
			t1 = temp;
		}

		if(t1 < 0)
		{
			
		}
		else if(t0 < 0)
		{
			t = t1;
		}
		else
		{
			t = t0;
		}
	}

	if(t < 0)
	{
		dest[pixel] = make_uchar4(255, 255, 255, 255);
	}
	else
	{
		dest[pixel] = make_uchar4(100, 0, 100, 255);
	}
}


void RunRayTracer(uchar4* dest, const int imageW, const int imageH, const int xThreadsPerBlock, const float4 a_vCameraLocation, const float a_fNearPlaneDistance)
{
	dim3 numThreads(20, 20);
	dim3 numBlocks(64, 36);

	float4 cameraUp, cameraForward, cameraRight;
	float2 viewSize;

	cameraUp = make_float4(0, 1, 0, 0);
	cameraForward = make_float4(0, 0, 1, 0);
	cameraRight = make_float4(1, 0, 0, 0);
	viewSize = make_float2(imageW, imageH);

	RayTracer<<<numBlocks, numThreads>>>(dest, imageW, imageH, a_vCameraLocation, cameraUp, cameraForward, cameraRight, a_fNearPlaneDistance, viewSize);
}