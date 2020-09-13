#pragma once

#include <vector>
#include <string>
#include "vectors.h"
#include "hash.h"
#include "boundingbox.h"
#include "argparser.h"
#include "GL/glew.h"
#include "GL/glut.h"

class Vertex;
class Edge;
class Triangle;


struct VBOVert
{
	VBOVert() {}
	VBOVert(const Vec3f& p) {
		x = p.x();
		y = p.y();
		z = p.z();
	}
	float x, y, z;
};


struct VBOEdge
{
	VBOEdge() {}
	VBOEdge(unsigned int a, unsigned int b) {
		verts[0] = a;
		verts[1] = b;
	}
	unsigned int verts[2];
};


struct VBOTriVert
{
	VBOTriVert() {}
	VBOTriVert(const Vec3f& p, const Vec3f& n) {
		x = p.x();
		y = p.y();
		z = p.z();
		nx = n.x();
		ny = n.y();
		nz = n.z();
	}
	float x, y, z;
	float nx, ny, nz;
};


struct VBOTri
{
	VBOTri() {}
	VBOTri(unsigned int a, unsigned int b, unsigned int c) {
		verts[0] = a;
		verts[1] = b;
		verts[2] = c;
	}
	unsigned int verts[3];
};


class Mesh
{
public:
	Mesh(ArgParser* a) { args = a; }
	~Mesh();

	void Load(const std::string& input_file);

	int numVertices() const { return vertices.size(); }

	Vertex* addVertex(const Vec3f& pos);

	Vertex* getVertex(int i) const {
		assert(i >= 0 && numVertices());
		Vertex* v = vertices[i];
		assert(v != NULL);
		return v;
	}

	void setParentsChild(Vertex* p1, Vertex* p2, Vertex* child);

	Vertex* getChildVertex(Vertex* p1, Vertex* p2) const;

	int numEdges() const { return edges.size(); }

	Edge* getMeshEdge(Vertex* a, Vertex* b) const;

	int numTriangles() const { return triangles.size(); }

	void addTriangle(Vertex* a, Vertex* b, Vertex* c);

	void removeTriangle(Triangle* t);

	const BoundingBox& getBoundingBox() const { return bbox; }

	void initializeVBOs();

	void setupVBOs();

	void drawVBOs();

	void cleanupVBOs();

	void LoopSubdivision();

	void BufferflySubdevision();

	void TestLoopSubdivision();

	void TestBufferflySubdivision();

	void Simplefication(int target_tri_count);

private:

	Mesh(const Mesh&) { assert(0); exit(0); }
	const Mesh& operator=(const Mesh&) { assert(0); exit(0); }

	void setupTriVBOs();
	void setupEdgeVBOs();

	ArgParser* args;
	std::vector<Vertex*> vertices;
	edgeshashtype edges;
	triangleshashtype triangles;
	BoundingBox bbox;
	vphashtype vertex_parents;

	int num_boundary_edges;
	int num_crease_edges;
	int num_other_edges;

	GLuint mesh_tri_verts_VBO;
	GLuint mesh_tri_indices_VBO;
	GLuint mesh_verts_VBO;
	GLuint mesh_boundary_edge_indices_VBO;
	GLuint mesh_crease_edge_indices_VBO;
	GLuint mesh_other_edge_indices_VBO;

	void addEdgeAnchor();
	void moveOldVertices();
	void splitTriangles();

	void interpolatePoints();

	Vertex* addLoopAnchor(Edge* e);
	void moveLoopVertex(Vertex* p, Edge* e);

	void getNeighbors(Edge* e, std::vector<Vertex*>& neighbors);
	Vertex* bufferflyCaseA(
		const Edge* e,
		const std::vector<Vertex*>& neighbors_a,
		const std::vector<Vertex*>& neighbors_b);
	Vertex* bufferflyCaseB(
		const Edge* e,
		const std::vector<Vertex*>& neighbors_a,
		const std::vector<Vertex*>& neighbors_b);
	Vertex* bufferflyCaseC(
		const Edge* e, 
		const std::vector<Vertex*>& neighbors_a,
		const std::vector<Vertex*>& neighbors_b);
	Vertex* bufferflyCaseD(const Edge* e);
	Vec3f getWeightVectorForCaseB(
		const Vertex* p,
		const std::vector<Vertex*>& neighbors);
};