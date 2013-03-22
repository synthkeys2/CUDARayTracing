#ifndef _CUDA_RAY_TRACER_h_
#define _CUDA_RAY_TRACER_h_

//OPENGL CONFIGURATION

//Window Settings
//#define FULLSCREEN
#define WINDOW_HEIGHT		720
#define WINDOW_WIDTH		1280
#define WINDOW_POS_X		0
#define WINDOW_POS_Y		0

//Timer settings
#define REFRESH_DELAY 10

//MATHEMATICAL CONSTANTS
#define FLT_MIN         1.175494351e-38F
#define FLT_MAX         3.402823466e+38F
#define INFINITY		FLT_MAX

#include <vector_types.h>

extern "C" void RunRayTracer(uchar4* dest, const int imageW, const int imageH);


#endif