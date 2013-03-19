/*
    pbrt source code Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include "RayTracer.h"
#include <math.h>

class Point;
class Normal;

// Geometry Declarations
class Vector {
public:
    // Vector Public Methods
    Vector() { x = y = z = 0.f; }
    Vector(float xx, float yy, float zz)
        : x(xx), y(yy), z(zz) {
    }

    explicit Vector(const Point &p);
	explicit Vector(const Normal &n);
    
	Vector operator+(const Vector &v) const {
        return Vector(x + v.x, y + v.y, z + v.z);
    }
    
    Vector& operator+=(const Vector &v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    Vector operator-(const Vector &v) const {
        return Vector(x - v.x, y - v.y, z - v.z);
    }
    
    Vector& operator-=(const Vector &v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }
    Vector operator*(float f) const { return Vector(f*x, f*y, f*z); }
    
    Vector &operator*=(float f) {
        x *= f; y *= f; z *= f;
        return *this;
    }
    Vector operator/(float f) const {
        float inv = 1.f / f;
        return Vector(x * inv, y * inv, z * inv);
    }
    
    Vector &operator/=(float f) {
        float inv = 1.f / f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    Vector operator-() const { return Vector(-x, -y, -z); }
    float operator[](int i) const {
        return (&x)[i];
    }
    
    float &operator[](int i) {
        return (&x)[i];
    }
    float LengthSquared() const { return x*x + y*y + z*z; }
    float Length() const { return sqrtf(LengthSquared()); }

    bool operator==(const Vector &v) const {
        return x == v.x && y == v.y && z == v.z;
    }
    bool operator!=(const Vector &v) const {
        return x != v.x || y != v.y || z != v.z;
    }

    // Vector Public Data
    float x, y, z;
};


class Point {
public:
    // Point Public Methods
    Point() { x = y = z = 0.f; }
    Point(float xx, float yy, float zz)
        : x(xx), y(yy), z(zz) {
    }
    Point operator+(const Vector &v) const {
        return Point(x + v.x, y + v.y, z + v.z);
    }
    
    Point &operator+=(const Vector &v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    Vector operator-(const Point &p) const {
        return Vector(x - p.x, y - p.y, z - p.z);
    }
    
    Point operator-(const Vector &v) const {
        return Point(x - v.x, y - v.y, z - v.z);
    }
    
    Point &operator-=(const Vector &v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }
    Point &operator+=(const Point &p) {
        x += p.x; y += p.y; z += p.z;
        return *this;
    }
    Point operator+(const Point &p) const {
        return Point(x + p.x, y + p.y, z + p.z);
    }
    Point operator* (float f) const {
        return Point(f*x, f*y, f*z);
    }
    Point &operator*=(float f) {
        x *= f; y *= f; z *= f;
        return *this;
    }
    Point operator/ (float f) const {
        float inv = 1.f/f;
        return Point(inv*x, inv*y, inv*z);
    }
    Point &operator/=(float f) {
        float inv = 1.f/f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    float operator[](int i) const {
        return (&x)[i];
    }
    
    float &operator[](int i) {
        return (&x)[i];
    }

    bool operator==(const Point &p) const {
        return x == p.x && y == p.y && z == p.z;
    }
    bool operator!=(const Point &p) const {
        return x != p.x || y != p.y || z != p.z;
    }

    // Point Public Data
    float x, y, z;
};

class Normal {
public:
    // Normal Public Methods
    Normal() { x = y = z = 0.f; }
    Normal(float xx, float yy, float zz)
        : x(xx), y(yy), z(zz) {
    }
    Normal operator-() const {
        return Normal(-x, -y, -z);
    }
    Normal operator+ (const Normal &n) const {
        return Normal(x + n.x, y + n.y, z + n.z);
    }
    
    Normal& operator+=(const Normal &n) {
        x += n.x; y += n.y; z += n.z;
        return *this;
    }
    Normal operator- (const Normal &n) const {
        return Normal(x - n.x, y - n.y, z - n.z);
    }
    
    Normal& operator-=(const Normal &n) {
        x -= n.x; y -= n.y; z -= n.z;
        return *this;
    }
    Normal operator*(float f) const {
        return Normal(f*x, f*y, f*z);
    }
    
    Normal &operator*=(float f) {
        x *= f; y *= f; z *= f;
        return *this;
    }
    Normal operator/(float f) const {
        float inv = 1.f/f;
        return Normal(x * inv, y * inv, z * inv);
    }
    
    Normal &operator/=(float f) {
        float inv = 1.f/f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    float LengthSquared() const { return x*x + y*y + z*z; }
    float Length() const        { return sqrtf(LengthSquared()); }
    
#ifndef NDEBUG
    Normal(const Normal &n) {
        x = n.x; y = n.y; z = n.z;
    }
    
    Normal &operator=(const Normal &n) {
        x = n.x; y = n.y; z = n.z;
        return *this;
    }
#endif // !NDEBUG
    explicit Normal(const Vector &v)
      : x(v.x), y(v.y), z(v.z) {
    }
    float operator[](int i) const {
        return (&x)[i];
    }
    
    float &operator[](int i) {
        return (&x)[i];
    }

    bool operator==(const Normal &n) const {
        return x == n.x && y == n.y && z == n.z;
    }
    bool operator!=(const Normal &n) const {
        return x != n.x || y != n.y || z != n.z;
    }

    // Normal Public Data
    float x, y, z;
};


class Ray {
public:
    // Ray Public Methods
    Ray() : mint(0.f), maxt(INFINITY), time(0.f), depth(0) { }
    Ray(const Point &origin, const Vector &direction,
        float start, float end = INFINITY, float t = 0.f, int d = 0)
        : o(origin), d(direction), mint(start), maxt(end), time(t), depth(d) { }
    Ray(const Point &origin, const Vector &direction, const Ray &parent,
        float start, float end = INFINITY)
        : o(origin), d(direction), mint(start), maxt(end),
          time(parent.time), depth(parent.depth+1) { }
    Point operator()(float t) const { return o + d * t; }

    // Ray Public Data
    Point o;
    Vector d;
    mutable float mint, maxt;
    float time;
    int depth;
};



// Geometry Inline Functions
inline Vector::Vector(const Point &p)
    : x(p.x), y(p.y), z(p.z) {
}


inline Vector operator*(float f, const Vector &v) { return v*f; }
inline float Dot(const Vector &v1, const Vector &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}


inline float AbsDot(const Vector &v1, const Vector &v2) {
    return fabsf(Dot(v1, v2));
}




#endif // PBRT_CORE_GEOMETRY_H