#include <stdio.h>
#include <stdlib.h>

#include "RayTracer.h"
#include "geometry.h"

// OpenGL Graphics includes
#include <GL/glew.h>
#ifdef _WIN32
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

void InitializeOpenGL(int* argc, char** argv);
void Display();
void Reshape(int w, int h);
void ProcessKeyboard(unsigned char k, int, int);
void ProcessMouseClick(int button, int state, int x, int y);
void ProcessMouseMove(int x, int y);
void TimerEvent(int value);


int main(int argc, char** argv)
{
	InitializeOpenGL(&argc, argv);
	glutMainLoop();
    runTest(argc, argv);
}

void InitializeOpenGL(int* argc, char** argv)
{
	printf("Initializing GLUT");

    glutInit(argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(WINDOW_POS_X, WINDOW_POS_Y);
    glutCreateWindow(argv[0]);

	glutDisplayFunc(Display);
	glutKeyboardFunc(ProcessKeyboard);
	glutMouseFunc(ProcessMouseClick);
	glutMotionFunc(ProcessMouseMove);
	glutReshapeFunc(Reshape);
    glutTimerFunc(REFRESH_DELAY, TimerEvent, 0);

    printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));

    if (!glewIsSupported("GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        exit(EXIT_SUCCESS);
    }

    printf("OpenGL window created.\n");	
}

void Display()
{

}

void Reshape(int w, int h)
{

}

void ProcessKeyboard(unsigned char k, int, int)
{
   switch (k)
    {
        case '\033':
        case 'q':
        case 'Q':
            printf("Shutting down...\n");
            exit(EXIT_SUCCESS);
            break;
   }
}

void ProcessMouseClick(int button, int state, int x, int y)
{

}

void ProcessMouseMove(int x, int y)
{

}

void TimerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, TimerEvent, 0);
}