#include "geometry.h"

class HitInfo
{
public:
	Point HitLocation;
	Normal HitNormal;
};

class Shape
{
public:
	Shape();
	~Shape();

	virtual bool Intersect(Ray* ray, HitInfo* hitInfo) = 0;

private:
	Point mLocation;
};