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
	const int pixelIndex = imageW * iy + ix;
	float4 pixelColor;

	// Compute the center of the near plane. All rays will be computed as an offset from this point
	const float4 lookAt = cameraLocation + cameraForward * nearPlaneDistance;

	// Find where the ray intersects the near plane and create the vector portion of the ray from that
	const float4 rayMidPoint = lookAt + cameraRight * ((float(ix) / float(imageW) - 0.5f) * viewSize.x) + cameraUp * ((/*1 - */(float(iy) / float(imageH)) - 0.5f) * viewSize.y); 
	const float4 ray = normalize(rayMidPoint - cameraLocation);

	// Hardcoded sphere
	const float4 sphereCenter = make_float4(0, 0, 40, 1);
	const float radius = 1.0f;

	// Hardcoded light
	//const float4 lightPosition = make_float4(0, 20, 20, 1);
	const float4 lightPosition = make_float4(0, 0, -40, 1);

	// Calculate the three coefficients in the quadratic equation
	const float4 rayOriginMinusSphereCenter = cameraLocation - sphereCenter;

	const float A = dot(ray, ray);
	const float B = 2 * dot(rayOriginMinusSphereCenter, ray);
	const float C = dot(rayOriginMinusSphereCenter, rayOriginMinusSphereCenter) - radius * radius;

	//Calculate the discriminant
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
		pixelColor = make_float4(0.2f, 0.2f, 0.4f, 1.0f);
	}
	else
	{
		pixelColor = make_float4(0.4f, 0, 0.4f, 1.0f);

		const float4 intersectionPoint = cameraLocation + t * ray;
		const float4 intersectionNormal = normalize(intersectionPoint - sphereCenter);
		const float4 lightDirection = normalize(lightPosition - intersectionPoint);
		const float4 halfVector = normalize(lightDirection + normalize(cameraLocation - intersectionPoint));
		float diffuseStrength = dot(intersectionNormal, lightDirection);
		float specularStrength = dot(intersectionNormal, halfVector);
		diffuseStrength = clamp(diffuseStrength, 0.0f, 1.0f);
		specularStrength = clamp(specularStrength, 0.0f, 1.0f);
		specularStrength = pow(specularStrength, 15);
		float lightCoefficient = diffuseStrength + AMBIENT_STRENGTH;
		pixelColor.x = clamp(pixelColor.x * lightCoefficient + specularStrength, 0.0f, 1.0f);
		pixelColor.y = clamp(pixelColor.y * lightCoefficient + specularStrength, 0.0f, 1.0f);
		pixelColor.z = clamp(pixelColor.z * lightCoefficient + specularStrength, 0.0f, 1.0f);
	}

	uchar4 pixelColorBytes;
	pixelColorBytes.x = (unsigned char)(pixelColor.x * 255);
	pixelColorBytes.y = (unsigned char)(pixelColor.y * 255);
	pixelColorBytes.z = (unsigned char)(pixelColor.z * 255);
	pixelColorBytes.w = 255;
	dest[pixelIndex] = pixelColorBytes;
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