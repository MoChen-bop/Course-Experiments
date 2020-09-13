#pragma once
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "shader.h"
#include "loader.h"
#include "controller.h"

class Grass
{
public:
	Grass() {}
	Grass(float size, glm::vec3 position, float yaw);
	void draw(GLFWwindow* window);
private:
	Shader shader;
	GLuint vao;
	GLuint texture;
	int indexCount;
	glm::mat4 model_matrix;
};