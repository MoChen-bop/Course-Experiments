#pragma once

#include <vector>
#include "GL/glew.h"
#include "glm/glm.hpp"

#include "controller.h"

class Vertex
{
public:
	Vertex() : pos(glm::vec3(0.0f, 0.0f, 0.0f)), texCoord(glm::vec2(0.0f, 0.0f)) {}
	Vertex(const glm::vec3& _pos) : pos(_pos), texCoord(glm::vec2(abs(_pos.x), abs(_pos.y))) {}
	Vertex(const glm::vec3& _pos, const glm::vec2& _texCoord) : pos(_pos), texCoord(_texCoord) {}

	inline glm::vec3* GetPos() { return &pos; }
	inline glm::vec2* GetTexCoord() { return &texCoord; }
private:
	glm::vec3 pos;
	glm::vec2 texCoord;
};

class PinwheelMesh
{
public:
	PinwheelMesh();
	~PinwheelMesh();

	void DrawFan();
	void DrawBar();
	void DrawScrew();
	void spin(Controller controller, GLFWwindow* window);
private:
	enum
	{
		POSITION_VB,
		TEXCOORD_VB,
		NUM_BUFFERS
	};
	enum
	{
		FAN_VB,
		BAR_VB,
		SCREW_VB,
		NUM_COMPONENT
	};
	GLuint vertexArrayObject;
	GLuint vertexArrayBuffers[NUM_BUFFERS];
	Vertex* vertices[NUM_COMPONENT];
	int numVertices[NUM_COMPONENT];
	int totalNumVertices;

	Vertex* generateFanVertices();
	Vertex* generateBarVertices();
	Vertex* generateScrewVertices();
	std::vector<glm::vec3> positions;
	std::vector<glm::vec2> texCoords;
};