#include "edge.h"
#include "vertex.h"

Edge::Edge(Vertex* vs, Vertex* ve, Triangle* t)
{
	start_vertex = vs;
	end_vertex = ve;
	triangle = t;
	next = NULL;
	opposite = NULL;
	crease = 0;
}

Edge::~Edge()
{
	if (opposite != NULL)
		opposite->opposite = NULL;
}

float Edge::Length() const
{
	Vec3f diff = start_vertex->getPos() - end_vertex->getPos();
	return diff.Length();
}
