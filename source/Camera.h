#include "geometry.h"

class Camera
{
public:
	Camera();
	~Camera();

	Ray CreateRayThroughPixel(int x, int y);

private:
	Point mLocation;
	Point mLookAt;

	Vector mUp;
	Vector mForward;

	float mFocalLength;
	float mViewWidth;
	float mViewHeight;
	float mAspectRatio;
};