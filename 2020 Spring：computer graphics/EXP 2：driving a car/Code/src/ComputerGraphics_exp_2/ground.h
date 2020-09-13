#pragma once
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "shader.h"
#include "loader.h"
#include "controller.h"

class Ground
{
public:
	Ground(float SIZE);
	static void change();
	void draw(GLFWwindow* window, Camera camera, Car car, bool control_car);
private:
	Shader shader;
	GLuint vao;
	GLuint texture[3];
	int indexCount;
	glm::vec3 position;
	float yaw;
	glm::vec3 front;
};