#include "geometry.h"

class Camera
{
public:
	Camera();
	Camera(Point location, Point lookAt, Vector up, float focalLength, float viewWidth, float viewHeight, float aspectRatio);
	~Camera();

	Ray CreateRayThroughPixel(int x, int y);

private:
	Point mLocation;

	Vector mUp;
	Vector mForward;

	float mFocalLength;
	float mViewWidth;
	float mViewHeight;
	float mAspectRatio;
};