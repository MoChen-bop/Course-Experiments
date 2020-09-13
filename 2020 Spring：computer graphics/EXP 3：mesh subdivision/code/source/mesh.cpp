#include "mesh.h"
#include "glCanvas.h"
#include <fstream>
#include <sstream>
#include <math.h>
#include <windows.h>
#include "mesh.h"
#include "edge.h"
#include "vertex.h"
#include "triangle.h"


int Triangle::next_triangle_id = 0;

#define BUFFER_OFFSET(i) ((char *)NULL + (i))


Mesh::~Mesh()
{
	cleanupVBOs();

	std::vector<Triangle*> todo;
	for (triangleshashtype::iterator iter = triangles.begin();
		iter != triangles.end(); iter++) {
		Triangle* t = iter->second;
		todo.push_back(t);
	}
	int num_triangles = todo.size();
	for (int i = 0; i < num_triangles; i++) {
		removeTriangle(todo[i]);
	}

	int num_vertices = numVertices();
	for (int i = 0; i < num_vertices; i++) {
		delete vertices[i];
	}
}


Vertex* Mesh::addVertex(const Vec3f& position)
{
	int index = numVertices();

	Vertex* v = new Vertex(index, position);
	vertices.push_back(v);
	if (numVertices() == 1)
		bbox = BoundingBox(position, position);
	else
		bbox.Extend(position);
	return v;
}


void Mesh::addTriangle(Vertex* a, Vertex* b, Vertex* c)
{
	Triangle* t = new Triangle();

	Edge* ea = new Edge(a, b, t);
	Edge* eb = new Edge(b, c, t);
	Edge* ec = new Edge(c, a, t);

	t->setEdge(ea);

	ea->setNext(eb);
	eb->setNext(ec);
	ec->setNext(ea);

	assert(edges.find(std::make_pair(a, b)) == edges.end());
	assert(edges.find(std::make_pair(b, c)) == edges.end());
	assert(edges.find(std::make_pair(c, a)) == edges.end());

	edges[std::make_pair(a, b)] = ea;
	edges[std::make_pair(b, c)] = eb;
	edges[std::make_pair(c, a)] = ec;

	edgeshashtype::iterator ea_op = edges.find(std::make_pair(b, a));
	edgeshashtype::iterator eb_op = edges.find(std::make_pair(c, b));
	edgeshashtype::iterator ec_op = edges.find(std::make_pair(a, c));

	if (ea_op != edges.end()) { ea_op->second->setOpposite(ea); }
	if (eb_op != edges.end()) { eb_op->second->setOpposite(eb); }
	if (ec_op != edges.end()) { ec_op->second->setOpposite(ec); }

	assert(triangles.find(t->getID()) == triangles.end());
	triangles[t->getID()] = t;
}


void Mesh::removeTriangle(Triangle* t)
{
	Edge* ea = t->getEdge();
	Edge* eb = ea->getNext();
	Edge* ec = eb->getNext();
	Vertex* a = ea->getStartVertex();
	Vertex* b = eb->getStartVertex();
	Vertex* c = ec->getStartVertex();

	edges.erase(std::make_pair(a, b));
	edges.erase(std::make_pair(b, c));
	edges.erase(std::make_pair(c, a));
	triangles.erase(t->getID());

	delete ea;
	delete eb;
	delete ec;
	delete t;
}


Edge* Mesh::getMeshEdge(Vertex* a, Vertex* b) const
{
	edgeshashtype::const_iterator iter = edges.find(std::make_pair(a, b));
	if (iter == edges.end()) return NULL;
	return iter->second;
}


Vertex* Mesh::getChildVertex(Vertex* p1, Vertex* p2) const
{
	vphashtype::const_iterator iter = vertex_parents.find(std::make_pair(p1, p2));
	if (iter == vertex_parents.end()) return NULL;
	return iter->second;
}

void Mesh::setParentsChild(Vertex* p1, Vertex* p2, Vertex* child)
{
	assert(vertex_parents.find(std::make_pair(p1, p2)) == vertex_parents.end());
	vertex_parents[std::make_pair(p1, p2)] = child;
}

# define MAX_CHAR_PER_LINE 200

void Mesh::Load(const std::string& input_file)
{
	std::ifstream istr(input_file.c_str());
	if (!istr) {
		std::cout << "ERROR! CANNOT OPEN: " << input_file << std::endl;
		return;
	}

	char line[MAX_CHAR_PER_LINE];
	std::string token, token2;
	float x, y, z;
	int a, b, c;
	int index = 0;
	int vert_count = 0;
	int vert_index = 1;

	while (istr.getline(line, MAX_CHAR_PER_LINE)) {
		std::stringstream ss;
		ss << line;

		token = "";
		ss >> token;

		if (token == "") continue;

		if (token == std::string("usemt1") || token == std::string("g")) {
			vert_index = 1;
			index++;
		}
		else if (token == std::string("v")) {
			vert_count++;
			ss >> x >> y >> z;
			addVertex(Vec3f(x, y, z));
		}
		else if (token == std::string("f")) {
			a = b = c = -1;
			ss >> a >> b >> c;
			a -= vert_index;
			b -= vert_index;
			c -= vert_index;

			assert(a >= 0 && a < numVertices());
			assert(b >= 0 && b < numVertices());
			assert(c >= 0 && c < numVertices());
			addTriangle(getVertex(a), getVertex(b), getVertex(c));
		}
		else if (token == std::string("e")) {
			a = b = -1;
			ss >> a >> b >> token2;

			assert(a >= 0 && a <= numVertices());
			assert(b >= 0 && b <= numVertices());

			if (token2 == std::string("inf")) x = 1000000;
			x = atof(token2.c_str());
			Vertex* va = getVertex(a);
			Vertex* vb = getVertex(b);
			Edge* ab = getMeshEdge(va, vb);
			Edge* ba = getMeshEdge(vb, va);
			assert(ab != NULL);
			assert(ba != NULL);
			ab->setCrease(x);
			ba->setCrease(x);
		}
		else if (token == std::string("vt")) {}
		else if (token == std::string("vn")) {}
		else if (token[0] == '#') {}
		else {
			printf("LINE: '%s'", line);
		}
	}
}


Vec3f ComputeNormal(const Vec3f& p1, const Vec3f& p2, const Vec3f& p3)
{
	Vec3f v12 = p2;
	v12 -= p1;
	Vec3f v23 = p3;
	v23 -= p2;
	Vec3f normal;
	Vec3f::Cross3(normal, v12, v23);
	normal.Normalize();
	return normal;
}


void Mesh::initializeVBOs()
{
	glGenBuffers(1, &mesh_tri_verts_VBO);
	glGenBuffers(1, &mesh_tri_indices_VBO);
	glGenBuffers(1, &mesh_verts_VBO);
	glGenBuffers(1, &mesh_boundary_edge_indices_VBO);
	glGenBuffers(1, &mesh_crease_edge_indices_VBO);
	glGenBuffers(1, &mesh_other_edge_indices_VBO);
	setupVBOs();
}


void Mesh::setupVBOs()
{
	HandleGLError("in setup mesh VBOs");
	setupTriVBOs();
	setupEdgeVBOs();
	HandleGLError("leaving setup mesh");
}


void Mesh::setupTriVBOs()
{
	VBOTriVert* mesh_tri_verts;
	VBOTri* mesh_tri_indices;
	unsigned int num_tris = triangles.size();

	mesh_tri_verts = new VBOTriVert[num_tris * 3];
	mesh_tri_indices = new VBOTri[num_tris];

	unsigned int i = 0;
	triangleshashtype::iterator iter = triangles.begin();
	for (; iter != triangles.end(); iter++, i++) {
		Triangle* t = iter->second;
		Vec3f a = (*t)[0]->getPos();
		Vec3f b = (*t)[1]->getPos();
		Vec3f c = (*t)[2]->getPos();

		if (args->gouraud) {
			Vec3f normal = ComputeNormal(a, b, c);
			mesh_tri_verts[i * 3] = VBOTriVert(a, normal);
			mesh_tri_verts[i * 3 + 1] = VBOTriVert(b, normal);
			mesh_tri_verts[i * 3 + 2] = VBOTriVert(c, normal);
		}
		else {
			Vec3f normal = ComputeNormal(a, b, c);
			mesh_tri_verts[i * 3] = VBOTriVert(a, normal);
			mesh_tri_verts[i * 3 + 1] = VBOTriVert(b, normal);
			mesh_tri_verts[i * 3 + 2] = VBOTriVert(c, normal);
		}
		mesh_tri_indices[i] = VBOTri(i * 3, i * 3 + 1, i * 3 + 2);
	}
	glDeleteBuffers(1, &mesh_tri_verts_VBO);
	glDeleteBuffers(1, &mesh_tri_indices_VBO);

	glBindBuffer(GL_ARRAY_BUFFER, mesh_tri_verts_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VBOTriVert) * num_tris * 3, mesh_tri_verts, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_tri_indices_VBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(VBOTri) * num_tris, mesh_tri_indices, GL_STATIC_DRAW);

	delete[] mesh_tri_verts;
	delete[] mesh_tri_indices;
}


void Mesh::setupEdgeVBOs()
{
	VBOVert* mesh_verts;
	VBOEdge* mesh_boundary_edge_indices;
	VBOEdge* mesh_crease_edge_indices;
	VBOEdge* mesh_other_edge_indices;

	mesh_boundary_edge_indices = NULL;
	mesh_crease_edge_indices = NULL;
	mesh_other_edge_indices = NULL;

	unsigned int num_verts = vertices.size();

	num_boundary_edges = 0;
	num_crease_edges = 0;
	num_other_edges = 0;
	for (edgeshashtype::iterator iter = edges.begin(); iter != edges.end(); iter++) {
		Edge* e = iter->second;
		int a = e->getStartVertex()->getIndex();
		int b = e->getEndVertex()->getIndex();
		if (e->getOpposite() == NULL) {
			num_boundary_edges++;
		}
		else {
			if (a < b) continue;
			if (e->getCrease() > 0) num_crease_edges++;
			else num_other_edges++;
		}
	}

	mesh_verts = new VBOVert[num_verts];
	if (num_boundary_edges > 0)
		mesh_boundary_edge_indices = new VBOEdge[num_boundary_edges];
	if (num_crease_edges > 0)
		mesh_crease_edge_indices = new VBOEdge[num_crease_edges];
	if (num_other_edges > 0)
		mesh_other_edge_indices = new VBOEdge[num_other_edges];

	for (unsigned int i = 0; i < num_verts; i++) {
		mesh_verts[i] = VBOVert(vertices[i]->getPos());
	}

	int bi = 0;
	int ci = 0;
	int oi = 0;
	for (edgeshashtype::iterator iter = edges.begin(); iter != edges.end(); iter++) {
		Edge* e = iter->second;
		int a = e->getStartVertex()->getIndex();
		int b = e->getEndVertex()->getIndex();
		if (e->getOpposite() == NULL) {
			mesh_boundary_edge_indices[bi++] = VBOEdge(a, b);
		}
		else {
			if (a < b) continue;
			if (e->getCrease() > 0)
				mesh_crease_edge_indices[ci++] = VBOEdge(a, b);
			else
				mesh_other_edge_indices[oi++] = VBOEdge(a, b);
		}
	}

	assert(bi == num_boundary_edges);
	assert(ci == num_crease_edges);
	assert(oi == num_other_edges);

	glDeleteBuffers(1, &mesh_verts_VBO);
	glDeleteBuffers(1, &mesh_boundary_edge_indices_VBO);
	glDeleteBuffers(1, &mesh_crease_edge_indices_VBO);
	glDeleteBuffers(1, &mesh_other_edge_indices_VBO);

	glBindBuffer(GL_ARRAY_BUFFER, mesh_verts_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VBOVert) * num_verts, mesh_verts, GL_STATIC_DRAW);

	if (num_boundary_edges > 0) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_boundary_edge_indices_VBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER,
			sizeof(VBOEdge) * num_boundary_edges,
			mesh_boundary_edge_indices, GL_STATIC_DRAW);
	}
	if (num_crease_edges > 0) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_crease_edge_indices_VBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER,
			sizeof(VBOEdge) * num_crease_edges,
			mesh_crease_edge_indices, GL_STATIC_DRAW);
	}
	if (num_other_edges > 0) {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_other_edge_indices_VBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER,
			sizeof(VBOEdge) * num_other_edges,
			mesh_other_edge_indices, GL_STATIC_DRAW);
	}

	delete[] mesh_verts;
	delete[] mesh_boundary_edge_indices;
	delete[] mesh_crease_edge_indices;
	delete[] mesh_other_edge_indices;
}


void Mesh::cleanupVBOs()
{
	glDeleteBuffers(1, &mesh_tri_verts_VBO);
	glDeleteBuffers(1, &mesh_tri_indices_VBO);
	glDeleteBuffers(1, &mesh_verts_VBO);
	glDeleteBuffers(1, &mesh_boundary_edge_indices_VBO);
	glDeleteBuffers(1, &mesh_crease_edge_indices_VBO);
	glDeleteBuffers(1, &mesh_other_edge_indices_VBO);
}

void Mesh::drawVBOs()
{
	HandleGLError("in draw mesh");

	Vec3f center; bbox.getCenter(center);
	float s = 1 / bbox.maxDim();
	glScalef(s, s, s);
	glTranslatef(-center.x(), -center.y(), -center.z());

	if (args->wireframe) {
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1.1, 4.0);
	}

	unsigned int num_tris = triangles.size();
	glColor3f(1, 1, 1);

	glBindBuffer(GL_ARRAY_BUFFER, mesh_tri_verts_VBO);

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(VBOTriVert), BUFFER_OFFSET(0));
	glEnableClientState(GL_NORMAL_ARRAY);
	glNormalPointer(GL_FLOAT, sizeof(VBOTriVert), BUFFER_OFFSET(12));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_tri_indices_VBO);

	glDrawElements(GL_TRIANGLES, num_tris * 3, GL_UNSIGNED_INT, BUFFER_OFFSET(0));

	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	if (args->wireframe) {
		glDisable(GL_POLYGON_OFFSET_FILL);
	}

	if (args->wireframe) {
		glDisable(GL_LIGHTING);
		
		glBindBuffer(GL_ARRAY_BUFFER, mesh_verts_VBO);

		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, sizeof(VBOVert), BUFFER_OFFSET(0));

		glLineWidth(3);
		glColor3f(1, 0, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_boundary_edge_indices_VBO);

		glDrawElements(GL_LINES, num_boundary_edges * 2, GL_UNSIGNED_INT, BUFFER_OFFSET(0));

		glLineWidth(3);
		glColor3f(1, 1, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_crease_edge_indices_VBO);
		glDrawElements(GL_LINES, num_boundary_edges * 2, GL_UNSIGNED_INT, BUFFER_OFFSET(0));

		glLineWidth(1);
		glColor3f(0, 0, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_other_edge_indices_VBO);
		glDrawElements(GL_LINES, num_other_edges * 2, GL_UNSIGNED_INT, BUFFER_OFFSET(0));
		glDisableClientState(GL_VERTEX_ARRAY);
	}

	HandleGLError("leaving draw VBOs");
}


void Mesh::LoopSubdivision()
{
	printf("Subdivide the mesh! (Default: Loop subdivision) \n");
	addEdgeAnchor();
	moveOldVertices();
	splitTriangles();
}

void Mesh::BufferflySubdevision()
{
	printf("Subdivide the mesh! (Buferfly subdivision) \n");
	interpolatePoints();
	splitTriangles();
}


void Mesh::TestLoopSubdivision()
{
	printf("Begin testing loop subdivision \n");
	printf("N (number of vertices of mesh): %d", vertices.size());
	double time = 0;
	double counts = 0;
	LARGE_INTEGER nFreq;
	LARGE_INTEGER nBeginTime;
	LARGE_INTEGER nEndTime;
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&nBeginTime);

	addEdgeAnchor();
	moveOldVertices();
	splitTriangles();

	QueryPerformanceCounter(&nEndTime);
	time = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;

	printf(", Comsumed time: %f ms\n", time * 1000);

}

void Mesh::TestBufferflySubdivision()
{
	printf("Begin testing bufferfly subdivision \n");
	printf("N (number of vertices of mesh): %d", vertices.size());
	double time = 0;
	double counts = 0;
	LARGE_INTEGER nFreq;
	LARGE_INTEGER nBeginTime;
	LARGE_INTEGER nEndTime;
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&nBeginTime);

	interpolatePoints();
	splitTriangles();

	QueryPerformanceCounter(&nEndTime);
	time = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) / (double)nFreq.QuadPart;

	printf(", Comsumed time: %f ms\n", time * 1000);
}

void Mesh::addEdgeAnchor()
{
	for (edgeshashtype::iterator iter = edges.begin(); iter != edges.end(); iter++) {
		Edge* e = iter->second;
		Vertex* p1 = e->getStartVertex();
		Vertex* p2 = e->getEndVertex();

		Vertex* anchor = getChildVertex(p1, p2);
		if (anchor == NULL) {
			anchor = addLoopAnchor(e);
			setParentsChild(p1, p2, anchor);
		}
	}
}

void Mesh::moveOldVertices()
{
	int n = numVertices();
	bool* done = new bool[n];
	for (int i = 0; i < n; i++) {
		done[i] = false;
	}

	triangleshashtype::iterator iter = triangles.begin();
	for (; iter != triangles.end(); iter++) {
		Triangle* t = iter->second;

		Vertex* p = (*t)[0];
		if (!done[p->getIndex()]) {
			moveLoopVertex(p, t->getEdge());
			done[p->getIndex()] = true;
		}

		p = (*t)[1];
		if (!done[p->getIndex()]) {
			moveLoopVertex(p, t->getEdge()->getNext());
			done[p->getIndex()] = true;
		}

		p = (*t)[2];
		if (!done[p->getIndex()]) {
			moveLoopVertex(p, t->getEdge()->getNext()->getNext());
			done[p->getIndex()] = true;
		}
	}

	delete[] done;
}

void Mesh::splitTriangles()
{
	std::vector<Triangle*> todo;
	triangleshashtype::iterator iter = triangles.begin();
	triangleshashtype::iterator end = triangles.end();
	for (; iter != end; iter++) {
		Triangle* t = iter->second;
		todo.push_back(t);
	}
	int num_triangles = todo.size();
	for (int i = 0; i < num_triangles; i++) {
		Triangle* t = todo[i];
		Vertex* p1 = (*t)[0];
		Vertex* p2 = (*t)[1];
		Vertex* p3 = (*t)[2];
		Vertex* anchor1 = getChildVertex(p1, p2);
		Vertex* anchor2 = getChildVertex(p2, p3);
		Vertex* anchor3 = getChildVertex(p3, p1);
		addTriangle(p1, anchor1, anchor3);
		addTriangle(p2, anchor2, anchor1);
		addTriangle(p3, anchor3, anchor2);
		addTriangle(anchor1, anchor2, anchor3);
		removeTriangle(t);
	}
}


Vertex* Mesh::addLoopAnchor(Edge* e)
{
	Edge* opposite = e->getOpposite();
	if (opposite == NULL) {
		Vec3f v1 = e->getStartVertex()->getPos();
		Vec3f v2 = e->getEndVertex()->getPos();
		Vec3f a = (v1 + v2) / 2.0;
		Vertex* anchor = addVertex(a);
		return anchor;
	}
	else {
		Vec3f v1 = e->getStartVertex()->getPos();
		Vec3f v2 = e->getNext()->getStartVertex()->getPos();
		Vec3f v3 = e->getNext()->getNext()->getStartVertex()->getPos();
		Vec3f v4 = opposite->getNext()->getEndVertex()->getPos();
		Vec3f a = (3 * v1 + 3 * v2 + v3 + v4) / 8.0;
		Vertex* anchor = addVertex(a);
		return anchor;
	}
}

void Mesh::moveLoopVertex(Vertex* p, Edge* e)
{
	std::vector<Vertex*> neighbors;
	Vertex* p1 = e->getStartVertex();
	assert(p1 == p);
	Vertex* p2 = e->getEndVertex();
	neighbors.push_back(getChildVertex(p1, p2));

	Edge* e_iter = e->getOpposite();
	while (e_iter != NULL && e_iter->getNext() != e) {
		e_iter = e_iter->getNext();
		Vertex* p1 = e_iter->getStartVertex();
		Vertex* p2 = e_iter->getEndVertex();
		Vertex* anchor = getChildVertex(p1, p2);
		neighbors.push_back(anchor);
		e_iter = e_iter->getOpposite();
	}

	if (e_iter == NULL) { // for crease and boundary
		Edge* e_inverse_iter = e->getNext()->getNext();
		while (e_inverse_iter->getOpposite() != NULL) {
			e_inverse_iter = e_inverse_iter->getOpposite()->getNext()->getNext();
		}

		Vertex* neighbor1 = neighbors[neighbors.size() - 1];
		Vertex* p1 = e_inverse_iter->getStartVertex();
		Vertex* p2 = e_inverse_iter->getEndVertex();
		Vertex* neighbor2 = getChildVertex(p1, p2);
		Vec3f new_v = p->getPos() * 3. / 4.;
		new_v += neighbor1->getPos() / 8. + neighbor2->getPos() / 8.;
		p->setPos(new_v);
	}
	else { // for interior
		int n = neighbors.size();
		assert(n != 0);

		double beta = (5. / 8. - pow(3. / 8. + cos(2 * M_PI / n) / 4., 2)) / n;
		Vec3f new_v = p->getPos() * (1 - n * beta);
		for (int i = 0; i < n; i++) {
			new_v += neighbors[i]->getPos() * beta;
		}
		p->setPos(new_v);
	}
}


void Mesh::interpolatePoints()
{
	for (edgeshashtype::iterator iter = edges.begin(); iter != edges.end(); iter++) {
		Edge* e = iter->second;
		Vertex* neighbor1a = e->getStartVertex();
		Vertex* neighbor1b = e->getEndVertex();

		if (getChildVertex(neighbor1a, neighbor1b) != NULL) {
			continue;
		}

		if (e->getOpposite() == NULL) { // for case (d), boundary and crease
			Vertex* interpolatedVertex = bufferflyCaseD(e);
			setParentsChild(neighbor1a, neighbor1b, interpolatedVertex);
		}
		else {
			std::vector<Vertex*> neighbors_a;
			getNeighbors(e, neighbors_a);

			std::vector<Vertex*> neighbors_b;
			getNeighbors(e->getOpposite(), neighbors_b);

			Vertex* interpolatedVertex = NULL;
			if (neighbors_a.size() == 6 && neighbors_b.size() == 6) {
				interpolatedVertex = bufferflyCaseA(e, neighbors_a, neighbors_b);
			}
			else if (neighbors_a.size() != 6 && neighbors_b.size() == 6) {
				interpolatedVertex = bufferflyCaseB(e, neighbors_a, neighbors_b);
			}
			else if (neighbors_a.size() == 6 && neighbors_b.size() != 6) {
				interpolatedVertex = bufferflyCaseB(e->getOpposite(), neighbors_b, neighbors_a);
			}
			else if (neighbors_a.size() != 6 && neighbors_b.size() != 6) {
				interpolatedVertex = bufferflyCaseC(e, neighbors_a, neighbors_b);
			}
			setParentsChild(neighbor1a, neighbor1b, interpolatedVertex);
		}
	}
}


void Mesh::getNeighbors(Edge* e, std::vector<Vertex*>& neighbors)
{
	Edge* e_iter = e;
	neighbors.push_back(e_iter->getEndVertex());
	while (e_iter->getOpposite() != NULL &&
		e_iter->getOpposite()->getNext() != e) {
		e_iter = e_iter->getOpposite()->getNext();
		neighbors.push_back(e_iter->getEndVertex());
	}

	if (e_iter->getOpposite() == NULL) {
		std::vector<Vertex*> _neighbors;
		e_iter = e->getNext()->getNext();
		_neighbors.push_back(e_iter->getStartVertex());
		while (e_iter->getOpposite() != NULL) {
			e_iter = e_iter->getOpposite()->getNext()->getNext();
			_neighbors.push_back(e_iter->getStartVertex());
		}
		for (int i = _neighbors.size() - 1; i >= 0; i--) {
			neighbors.push_back(_neighbors[i]);
		}
	}
}

Vertex* Mesh::bufferflyCaseA(const Edge* e, 
	const std::vector<Vertex*>& neighbors_a, const std::vector<Vertex*>& neighbors_b)
{
	assert(neighbors_a.size() == 6);
	assert(neighbors_b.size() == 6);

	Vertex* a = e->getStartVertex();
	Vertex* b = e->getEndVertex();

	Vec3f interpolated = (a->getPos() + b->getPos()) / 2.;
	interpolated += neighbors_a[1]->getPos() / 8.;
	interpolated -= neighbors_a[2]->getPos() / 16.;
	interpolated -= neighbors_a[4]->getPos() / 16.;

	interpolated += neighbors_b[1]->getPos() / 8.;
	interpolated -= neighbors_b[2]->getPos() / 16.;
	interpolated -= neighbors_b[4]->getPos() / 16.;

	Vertex* interpolatedVertex = addVertex(interpolated);
	return interpolatedVertex;
}

Vertex* Mesh::bufferflyCaseB(const Edge* e, 
	const std::vector<Vertex*>& neighbors_a, const std::vector<Vertex*>& neighbors_b)
{
	assert(neighbors_a.size() != 6);
	assert(neighbors_b.size() == 6);
	Vec3f interpolated = getWeightVectorForCaseB(e->getStartVertex(), neighbors_a);

	Vertex* interpolatedVertex = addVertex(interpolated);
	return interpolatedVertex;
}

Vertex* Mesh::bufferflyCaseC(const Edge* e, 
	const std::vector<Vertex*>& neighbors_a, const std::vector<Vertex*>& neighbors_b)
{
	assert(neighbors_a.size() != 6);
	assert(neighbors_b.size() != 6);
	Vec3f interpolated1 = getWeightVectorForCaseB(e->getStartVertex(), neighbors_a);
	Vec3f interpolated2 = getWeightVectorForCaseB(e->getEndVertex(), neighbors_b);
	Vec3f interpolated = (interpolated1 + interpolated2) / 2.;
	Vertex* interpolatedVertex = addVertex(interpolated);
	return interpolatedVertex;
}

Vertex* Mesh::bufferflyCaseD(const Edge* e)
{
	Vertex* neighbor1a = e->getStartVertex();
	Vertex* neighbor1b = e->getEndVertex();

	Edge* e_iter = e->getNext()->getNext();
	while (e_iter->getOpposite() != NULL) {
		e_iter = e_iter->getOpposite()->getNext()->getNext();
	}
	Vertex* neighbor2a = e_iter->getStartVertex();

	e_iter = e->getNext();
	while (e_iter->getOpposite() != NULL) {
		e_iter = e_iter->getOpposite()->getNext();
	}
	Vertex* neighbor2b = e_iter->getEndVertex();

	Vec3f interpolated = (neighbor1a->getPos() * 9. + neighbor1b->getPos() * 9.
		- neighbor2a->getPos() - neighbor2b->getPos()) / 16.;
	Vertex* interpolatedVertex = addVertex(interpolated);
	return interpolatedVertex;
}

Vec3f Mesh::getWeightVectorForCaseB(const Vertex* p, 
	const std::vector<Vertex*>& neighbors)
{
	assert(neighbors.size() != 6);

	Vec3f interpolated = p->getPos() * 3. / 4.;
	if (neighbors.size() == 3) {
		interpolated += neighbors[0]->getPos() * 5. / 12.;
		interpolated -= neighbors[1]->getPos() / 12.;
		interpolated -= neighbors[2]->getPos() / 12.;
	}
	else if (neighbors.size() == 4) {
		interpolated += neighbors[0]->getPos() * 3. / 8.;
		interpolated -= neighbors[2]->getPos() / 8.;
	}
	else if (neighbors.size() >= 5) {
		int k = neighbors.size();
		for (int i = 0; i < k; i++) {
			double weight = (1. / 4. + cos(2. * i * M_PI / k)
				+ 1. / 2. * cos(4. * i * M_PI / k) ) / k;
			interpolated += neighbors[i]->getPos() * weight;
		}
	}
	return interpolated;
}

void Mesh::Simplefication(int target_tri_count)
{
	vertex_parents.clear();

	printf("Simplify the mesh! %d -> %d\n", numTriangles(), target_tri_count);
}