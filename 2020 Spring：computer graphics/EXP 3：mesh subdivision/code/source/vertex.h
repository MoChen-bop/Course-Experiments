#pragma once

#include "vectors.h"

class Vertex
{
public:
	Vertex(int i, const Vec3f& pos) : position(pos) { index = i; }

	int getIndex() const { return index; }
	double x() const { return position.x(); }
	double y() const { return position.y(); }
	double z() const { return position.z(); }
	const Vec3f& getPos() const { return position; }

	void setPos(Vec3f v) { position = v; }

private:
	Vertex() { assert(0); exit(0); }
	Vertex(const Vertex&) { assert(0); exit(0); }
	Vertex& operator=(const Vertex&) { assert(0); exit(0); }

	Vec3f position;
	int index;
};