#ifndef _CUDA_RAY_TRACER_h_
#define _CUDA_RAY_TRACER_h_

// OPENGL CONFIGURATION

// Window Settings
//#define FULLSCREEN
#define WINDOW_HEIGHT		720
#define WINDOW_WIDTH		1280
#define WINDOW_POS_X		0
#define WINDOW_POS_Y		0

// Timer settings
#define REFRESH_DELAY 10

// MATHEMATICAL CONSTANTS
#define FLT_MIN			1.175494351e-38F
#define FLT_MAX			3.402823466e+38F
#define INFINITY		FLT_MAX

// INITIAL CAMERA VALUES
#define CAMERA_LOCATION				0, 0, 0, 1
#define NEAR_PLANE_DISTANCE			1500.0f
#define CAMERA_MOVEMENT_DELTA		0.2f
#define NEAR_PLANE_MOVEMENT_DELTA	10.0f

// LIGHT CONSTANTS
#define AMBIENT_STRENGTH			0.25f
#define BACKGROUND_COLOR			0.2f, 0.2f, 0.4f, 1.0f

#include <vector_types.h>

extern "C" void RunRayTracer(uchar4* dest, const int imageW, const int imageH, const int xThreadsPerBlock, float4 a_vCameraLocation, float a_fNearPlaneDistance);


#endif