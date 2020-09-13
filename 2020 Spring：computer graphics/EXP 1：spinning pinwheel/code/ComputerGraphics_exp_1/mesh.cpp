#include "mesh.h"

#include <vector>
#include "glm/gtc/matrix_transform.hpp"

PinwheelMesh::PinwheelMesh()
{
	vertices[FAN_VB] = generateFanVertices();
	vertices[BAR_VB] = generateBarVertices();
	vertices[SCREW_VB] = generateScrewVertices();

	glGenVertexArrays(1, &vertexArrayObject);
	glBindVertexArray(vertexArrayObject);

	for (unsigned int i = 0; i < numVertices[FAN_VB]; i++) {
		positions.push_back(*vertices[FAN_VB][i].GetPos());
		texCoords.push_back(*vertices[FAN_VB][i] .GetTexCoord());
	}
	for (unsigned int i = 0; i < numVertices[BAR_VB]; i++) {
		positions.push_back(*vertices[BAR_VB][i].GetPos());
		texCoords.push_back(*vertices[BAR_VB][i].GetTexCoord());
	}
	for (unsigned int i = 0; i < numVertices[SCREW_VB]; i++) {
		positions.push_back(*vertices[SCREW_VB][i].GetPos());
		texCoords.push_back(*vertices[SCREW_VB][i].GetTexCoord());
	}

	totalNumVertices = numVertices[FAN_VB] + numVertices[BAR_VB] + numVertices[SCREW_VB];
	glGenBuffers(NUM_BUFFERS, vertexArrayBuffers);
	glBindBuffer(GL_ARRAY_BUFFER, vertexArrayBuffers[POSITION_VB]);
	glBufferData(GL_ARRAY_BUFFER, totalNumVertices * sizeof(positions[0]), &positions[0], GL_STREAM_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vertexArrayBuffers[TEXCOORD_VB]);
	glBufferData(GL_ARRAY_BUFFER, totalNumVertices * sizeof(texCoords[0]), &texCoords[0], GL_STREAM_DRAW);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindVertexArray(0);
}

PinwheelMesh::~PinwheelMesh()
{
	glDeleteVertexArrays(1, &vertexArrayObject);
	delete[] vertices[FAN_VB];
	delete[] vertices[BAR_VB];
}

void PinwheelMesh::DrawFan()
{
	glBindVertexArray(vertexArrayObject);
	glDrawArrays(GL_TRIANGLES, 0, numVertices[FAN_VB]);
	glBindVertexArray(0);
}

void PinwheelMesh::DrawBar()
{
	glBindVertexArray(vertexArrayObject);
	glDrawArrays(GL_TRIANGLES, numVertices[FAN_VB], numVertices[FAN_VB] + numVertices[BAR_VB]);
	glBindVertexArray(0);
}

void PinwheelMesh::DrawScrew()
{
	glBindVertexArray(vertexArrayObject);
	glDrawArrays(GL_TRIANGLES, numVertices[FAN_VB] + numVertices[BAR_VB], numVertices[FAN_VB] + numVertices[BAR_VB] + numVertices[SCREW_VB]);
	glBindVertexArray(0);
}

void PinwheelMesh::spin(Controller controller, GLFWwindow* window)
{
	controller.updateMatrix(window);
	for (int i = 0; i < numVertices[FAN_VB]; i++) {
		glm::vec4 temp = controller.getMatrix() * glm::vec4(positions[i], 1.0f);
		positions[i].x = temp.x;
		positions[i].y = temp.y;
		positions[i].z = temp.z;
	}
	glBindBuffer(GL_ARRAY_BUFFER, vertexArrayBuffers[POSITION_VB]);
	glBufferData(GL_ARRAY_BUFFER, totalNumVertices * sizeof(positions[0]), &positions[0], GL_STREAM_DRAW);
}

Vertex* PinwheelMesh::generateFanVertices()
{
	glm::vec4 x11(-0.1, 0.057735, 0.0, 1.0);
	glm::vec4 x12(0.1, 0.057735, 0.0, 1.0);
	glm::vec4 x13(0.0, 0.057735 + 0.3, 0.0, 1.0);
	glm::vec4 x14(0.2, 0.057735 + 0.3, 0.0, 1.0);
	glm::vec4 x15(-0.1, 0.057735 + 0.3 + 0.15, 0.0, 1.0);

	glm::mat4 temp = glm::mat4(1.0f);
	glm::mat4 spinMatrix = glm::rotate(temp, glm::radians(120.0f), glm::vec3(0.0f, 0.0f, 1.0f));

	glm::vec4 x21 = spinMatrix * x11;
	glm::vec4 x22 = spinMatrix * x12;
	glm::vec4 x23 = spinMatrix * x13;
	glm::vec4 x24 = spinMatrix * x14;
	glm::vec4 x25 = spinMatrix * x15;

	glm::vec4 x31 = spinMatrix * x21;
	glm::vec4 x32 = spinMatrix * x22;
	glm::vec4 x33 = spinMatrix * x23;
	glm::vec4 x34 = spinMatrix * x24;
	glm::vec4 x35 = spinMatrix * x25;

	float thick = 0.02;

	Vertex* vertices = new Vertex[132] {
		Vertex(glm::vec3(x11.x, x11.y, thick)), Vertex(glm::vec3(x21.x, x21.y, thick)), Vertex(glm::vec3(x31.x, x31.y, thick)),
		Vertex(glm::vec3(x11.x, x11.y, thick)), Vertex(glm::vec3(x12.x, x12.y, thick)), Vertex(glm::vec3(x13.x, x13.y, thick)),
		Vertex(glm::vec3(x12.x, x12.y, thick)), Vertex(glm::vec3(x14.x, x14.y, thick)), Vertex(glm::vec3(x13.x, x13.y, thick)),
		Vertex(glm::vec3(x13.x, x13.y, thick)), Vertex(glm::vec3(x14.x, x14.y, thick)), Vertex(glm::vec3(x15.x, x15.y, thick)),
		Vertex(glm::vec3(x21.x, x21.y, thick)), Vertex(glm::vec3(x22.x, x22.y, thick)), Vertex(glm::vec3(x23.x, x23.y, thick)),
		Vertex(glm::vec3(x22.x, x22.y, thick)), Vertex(glm::vec3(x24.x, x24.y, thick)), Vertex(glm::vec3(x23.x, x23.y, thick)),
		Vertex(glm::vec3(x23.x, x23.y, thick)), Vertex(glm::vec3(x24.x, x24.y, thick)), Vertex(glm::vec3(x25.x, x25.y, thick)),
		Vertex(glm::vec3(x31.x, x31.y, thick)), Vertex(glm::vec3(x32.x, x32.y, thick)), Vertex(glm::vec3(x33.x, x33.y, thick)),
		Vertex(glm::vec3(x32.x, x32.y, thick)), Vertex(glm::vec3(x34.x, x34.y, thick)), Vertex(glm::vec3(x33.x, x33.y, thick)),
		Vertex(glm::vec3(x33.x, x33.y, thick)), Vertex(glm::vec3(x34.x, x34.y, thick)), Vertex(glm::vec3(x35.x, x35.y, thick)),

		Vertex(glm::vec3(x11.x, x11.y, - thick)), Vertex(glm::vec3(x21.x, x21.y, - thick)), Vertex(glm::vec3(x31.x, x31.y, - thick)),
		Vertex(glm::vec3(x11.x, x11.y, - thick)), Vertex(glm::vec3(x12.x, x12.y, - thick)), Vertex(glm::vec3(x13.x, x13.y, - thick)),
		Vertex(glm::vec3(x12.x, x12.y, - thick)), Vertex(glm::vec3(x14.x, x14.y, - thick)), Vertex(glm::vec3(x13.x, x13.y, - thick)),
		Vertex(glm::vec3(x13.x, x13.y, - thick)), Vertex(glm::vec3(x14.x, x14.y, - thick)), Vertex(glm::vec3(x15.x, x15.y, - thick)),
		Vertex(glm::vec3(x21.x, x21.y, - thick)), Vertex(glm::vec3(x22.x, x22.y, - thick)), Vertex(glm::vec3(x23.x, x23.y, - thick)),
		Vertex(glm::vec3(x22.x, x22.y, - thick)), Vertex(glm::vec3(x24.x, x24.y, - thick)), Vertex(glm::vec3(x23.x, x23.y, - thick)),
		Vertex(glm::vec3(x23.x, x23.y, - thick)), Vertex(glm::vec3(x24.x, x24.y, - thick)), Vertex(glm::vec3(x25.x, x25.y, - thick)),
		Vertex(glm::vec3(x31.x, x31.y, - thick)), Vertex(glm::vec3(x32.x, x32.y, - thick)), Vertex(glm::vec3(x33.x, x33.y, - thick)),
		Vertex(glm::vec3(x32.x, x32.y, - thick)), Vertex(glm::vec3(x34.x, x34.y, - thick)), Vertex(glm::vec3(x33.x, x33.y, - thick)),
		Vertex(glm::vec3(x33.x, x33.y, - thick)), Vertex(glm::vec3(x34.x, x34.y, - thick)), Vertex(glm::vec3(x35.x, x35.y, - thick)),

		Vertex(glm::vec3(x12.x, x12.y, thick)), Vertex(glm::vec3(x12.x, x12.y, - thick)), Vertex(glm::vec3(x14.x, x14.y, thick)),
		Vertex(glm::vec3(x12.x, x12.y, - thick)), Vertex(glm::vec3(x14.x, x14.y, - thick)), Vertex(glm::vec3(x14.x, x14.y, thick)),
		Vertex(glm::vec3(x14.x, x14.y, thick)), Vertex(glm::vec3(x14.x, x14.y, - thick)), Vertex(glm::vec3(x15.x, x15.y, thick)),
		Vertex(glm::vec3(x14.x, x14.y, - thick)), Vertex(glm::vec3(x15.x, x15.y, - thick)), Vertex(glm::vec3(x15.x, x15.y, thick)),
		Vertex(glm::vec3(x13.x, x13.y, thick)), Vertex(glm::vec3(x13.x, x13.y, - thick)), Vertex(glm::vec3(x15.x, x15.y, thick)),
		Vertex(glm::vec3(x13.x, x13.y, - thick)), Vertex(glm::vec3(x15.x, x15.y, - thick)), Vertex(glm::vec3(x15.x, x15.y, thick)),
		Vertex(glm::vec3(x11.x, x11.y, thick)), Vertex(glm::vec3(x11.x, x11.y, -thick)), Vertex(glm::vec3(x13.x, x13.y, thick)),
		Vertex(glm::vec3(x11.x, x11.y, -thick)), Vertex(glm::vec3(x13.x, x13.y, -thick)), Vertex(glm::vec3(x13.x, x13.y, thick)),

		Vertex(glm::vec3(x22.x, x22.y, thick)), Vertex(glm::vec3(x22.x, x22.y, -thick)), Vertex(glm::vec3(x24.x, x24.y, thick)),
		Vertex(glm::vec3(x22.x, x22.y, -thick)), Vertex(glm::vec3(x24.x, x24.y, -thick)), Vertex(glm::vec3(x24.x, x24.y, thick)),
		Vertex(glm::vec3(x24.x, x24.y, thick)), Vertex(glm::vec3(x24.x, x24.y, -thick)), Vertex(glm::vec3(x25.x, x25.y, thick)),
		Vertex(glm::vec3(x24.x, x24.y, -thick)), Vertex(glm::vec3(x25.x, x25.y, -thick)), Vertex(glm::vec3(x25.x, x25.y, thick)),
		Vertex(glm::vec3(x23.x, x23.y, thick)), Vertex(glm::vec3(x23.x, x23.y, -thick)), Vertex(glm::vec3(x25.x, x25.y, thick)),
		Vertex(glm::vec3(x23.x, x23.y, -thick)), Vertex(glm::vec3(x25.x, x25.y, -thick)), Vertex(glm::vec3(x25.x, x25.y, thick)),
		Vertex(glm::vec3(x21.x, x21.y, thick)), Vertex(glm::vec3(x21.x, x21.y, -thick)), Vertex(glm::vec3(x23.x, x23.y, thick)),
		Vertex(glm::vec3(x21.x, x21.y, -thick)), Vertex(glm::vec3(x23.x, x23.y, -thick)), Vertex(glm::vec3(x23.x, x23.y, thick)),

		Vertex(glm::vec3(x32.x, x32.y, thick)), Vertex(glm::vec3(x32.x, x32.y, -thick)), Vertex(glm::vec3(x34.x, x34.y, thick)),
		Vertex(glm::vec3(x32.x, x32.y, -thick)), Vertex(glm::vec3(x34.x, x34.y, -thick)), Vertex(glm::vec3(x34.x, x34.y, thick)),
		Vertex(glm::vec3(x34.x, x34.y, thick)), Vertex(glm::vec3(x34.x, x34.y, -thick)), Vertex(glm::vec3(x35.x, x35.y, thick)),
		Vertex(glm::vec3(x34.x, x34.y, -thick)), Vertex(glm::vec3(x35.x, x35.y, -thick)), Vertex(glm::vec3(x35.x, x35.y, thick)),
		Vertex(glm::vec3(x33.x, x33.y, thick)), Vertex(glm::vec3(x33.x, x33.y, -thick)), Vertex(glm::vec3(x35.x, x35.y, thick)),
		Vertex(glm::vec3(x33.x, x33.y, -thick)), Vertex(glm::vec3(x35.x, x35.y, -thick)), Vertex(glm::vec3(x35.x, x35.y, thick)),
		Vertex(glm::vec3(x31.x, x31.y, thick)), Vertex(glm::vec3(x31.x, x31.y, -thick)), Vertex(glm::vec3(x33.x, x33.y, thick)),
		Vertex(glm::vec3(x31.x, x31.y, -thick)), Vertex(glm::vec3(x33.x, x33.y, -thick)), Vertex(glm::vec3(x33.x, x33.y, thick)),
	
	};

	numVertices[FAN_VB] = 132;
	return vertices;
}

Vertex* PinwheelMesh::generateBarVertices()
{
	float h = 0.02;
	float H = - 0.7;
	float thick = 0.03;
	Vertex* vertices = new Vertex[36]{
		Vertex(glm::vec3(- h, h, thick)), Vertex(glm::vec3(h, h, thick)), Vertex(glm::vec3(h, H, thick)),
		Vertex(glm::vec3(- h, h, thick)), Vertex(glm::vec3(- h, H, thick)), Vertex(glm::vec3(h, H, thick)),
		Vertex(glm::vec3(-h, h, 2 * thick)), Vertex(glm::vec3(h, h, 2 * thick)), Vertex(glm::vec3(h, H, 2 * thick)),
		Vertex(glm::vec3(-h, h, 2 * thick)), Vertex(glm::vec3(-h, H, 2 * thick)), Vertex(glm::vec3(h, H, 2 * thick)),

		Vertex(glm::vec3(-h, h, thick)), Vertex(glm::vec3(h, h, thick)), Vertex(glm::vec3(h, h, 2 * thick)),
		Vertex(glm::vec3(- h, h, thick)), Vertex(glm::vec3(h, h, 2 * thick)), Vertex(glm::vec3(- h, h, 2 * thick)),
		Vertex(glm::vec3(-h, H, thick)), Vertex(glm::vec3(h, H, thick)), Vertex(glm::vec3(h, H, 2 * thick)),
		Vertex(glm::vec3(-h, H, thick)), Vertex(glm::vec3(h, H, 2 * thick)), Vertex(glm::vec3(-h, H, 2 * thick)),
		Vertex(glm::vec3(-h, h, thick)), Vertex(glm::vec3(- h, h, 2 * thick)), Vertex(glm::vec3(- h, H, thick)),
		Vertex(glm::vec3(-h, h, 2 *  thick)), Vertex(glm::vec3(- h, H, 2 * thick)), Vertex(glm::vec3(- h, H, thick)),
		Vertex(glm::vec3(h, h, thick)), Vertex(glm::vec3(h, h, 2 * thick)), Vertex(glm::vec3(h, H, thick)),
		Vertex(glm::vec3(h, h, 2 * thick)), Vertex(glm::vec3(h, H, 2 * thick)), Vertex(glm::vec3(h, H, thick)),
	};

	numVertices[BAR_VB] = 36;
	return vertices;
}

Vertex* PinwheelMesh::generateScrewVertices()
{
	float h = 0.005;
	float w = 0.08;
	float d = 0.02;

	Vertex* vertices = new Vertex[36]{
		Vertex(glm::vec3(-h, h, d)), Vertex(glm::vec3(h, h, d)), Vertex(glm::vec3(h, -h, d)),
		Vertex(glm::vec3(-h, h, d)), Vertex(glm::vec3(h, - h, d)), Vertex(glm::vec3(- h, -h, d)),
		Vertex(glm::vec3(-h, h, w)), Vertex(glm::vec3(h, h, w)), Vertex(glm::vec3(h, -h, w)),
		Vertex(glm::vec3(-h, h, w)), Vertex(glm::vec3(h, - h, w)), Vertex(glm::vec3(-h, -h, w)),

		Vertex(glm::vec3(-h, h, d)), Vertex(glm::vec3(- h, h, w)), Vertex(glm::vec3(- h, -h, w)),
		Vertex(glm::vec3(-h, h, d)), Vertex(glm::vec3(- h, - h, d)), Vertex(glm::vec3(- h, - h, w)),
		Vertex(glm::vec3(h, h, d)), Vertex(glm::vec3(h, h, w)), Vertex(glm::vec3(h, -h, w)),
		Vertex(glm::vec3(h, h, w)), Vertex(glm::vec3(h, -h, w)), Vertex(glm::vec3(h, -h, d)),
		Vertex(glm::vec3(-h, h, d)), Vertex(glm::vec3(-h, h, w)), Vertex(glm::vec3(h, h, w)),
		Vertex(glm::vec3(h, h, d)), Vertex(glm::vec3(- h, h, d)), Vertex(glm::vec3(h, h, w)),
		Vertex(glm::vec3(-h, - h, d)), Vertex(glm::vec3(-h, - h, w)), Vertex(glm::vec3(h, - h, w)),
		Vertex(glm::vec3(h, - h, d)), Vertex(glm::vec3(-h, - h, d)), Vertex(glm::vec3(h, - h, w)),
	};

	numVertices[SCREW_VB] = 36;

	return vertices;
}
