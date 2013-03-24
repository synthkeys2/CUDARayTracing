#include "Camera.h"

Camera::Camera()
	:	mLocation(Point(0, 10, 0)), mUp(Vector(0, 1, 0)), mFocalLength(0.1), mViewWidth(1), mViewHeight(0.75), mAspectRatio(1.33f) 
{
	mForward = Point(0, 0, 10) - mLocation;
}

Camera::Camera(Point location, Point lookAt, Vector up, float focalLength, float viewWidth, float viewHeight, float aspectRatio)
	:	mLocation(location), mUp(up), mFocalLength(focalLength), mViewWidth(viewWidth), mViewHeight(viewHeight), mAspectRatio(aspectRatio)
{
	mForward = lookAt - mLocation;
}

Camera::~Camera()
{

}

Ray Camera::CreateRayThroughPixel(int x, int y)
{

}