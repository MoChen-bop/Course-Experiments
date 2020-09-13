#include "grass.h"

Grass::Grass(float size, glm::vec3 position, float yaw)
{
	float s = size;
	std::vector<float> vertices = {
		0.0f, s, 0.0f,
		0.0f, 0.0f, 0.0f,
		s, 0.0f, 0.0f,
		s, s, 0.0f
	};
	std::vector<float> texCoords = {
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 0.0f
	};
	std::vector<unsigned int> indices = {
		0, 1, 2,    0, 2, 3
	};
	vao = Loader::getLoader()->loadVAO(vertices, indices, texCoords);
	indexCount = indices.size();

	model_matrix = glm::rotate(model_matrix, glm::radians(yaw), glm::vec3(0.0f, 1.0f, 0.0f));
	model_matrix = glm::translate(model_matrix, position);

	std::string texImage = "../../resources/textures/grass/grass.png";
	texture = Loader::getLoader()->loadTexture(texImage);
	shader = Shader("grass.vs", "grass.fs");
}

void Grass::draw(GLFWwindow* window)
{
	glDisable(GL_CULL_FACE);
	shader.use();
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	loadMatrix(window, shader, model_matrix);

	glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indexCount), GL_UNSIGNED_INT, (void*)0);

	glDisableVertexAttribArray(0);
	glBindVertexArray(0);
	glEnable(GL_CULL_FACE);
}
