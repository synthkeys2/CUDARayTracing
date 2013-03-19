#include <stdio.h>
#include "RayTracer.h"

void InitializeOpenGL(int* argc, char** argv);
void Display();
void Reshape(int w, int h);
void ProcessKeyboard(unsigned char k, int, int);
void ProcessMouseClick(int button, int state, int x, int y);
void ProcessMouseMove(int x, int y);


int main(int argc, char** argv)
{
	InitializeOpenGL(&argc, argv);
    runTest(argc, argv);
}

void InitializeOpenGL(int* argc, char** argv)
{
	printf("Initializing GLUT");

	
}