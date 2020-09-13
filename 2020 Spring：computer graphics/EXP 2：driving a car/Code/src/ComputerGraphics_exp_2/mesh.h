#pragma once

#include "glad/glad.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "shader.h"

#include <string>
#include <vector>

using namespace std;

struct Vertex 
{
	glm::vec3 Position;
	glm::vec3 Normal;
	glm::vec2 TexCoords;
	glm::vec3 Tangent;
	glm::vec3 Bitangent;
};

struct TextureInfo 
{
	unsigned int id;
	string type;
	string path;
};

class Mesh 
{
public:
	vector<Vertex>       vertices;
	vector<unsigned int> indices;
	vector<TextureInfo>      textures;
	unsigned int VAO;

	Mesh(vector<Vertex> vertices, vector<unsigned int> indices, vector<TextureInfo> textures);

	void Draw(Shader shader);

private:
	unsigned int VBO, EBO;
	void setupMesh();
};