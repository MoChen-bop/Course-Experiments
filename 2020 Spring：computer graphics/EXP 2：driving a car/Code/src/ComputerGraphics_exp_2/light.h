#pragma once
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "shader.h"
#include "camera.h"

class Light
{
public:
	glm::vec3 position;
	glm::vec3 front;
	float Yaw;
	float Pitch;

	Light();
	void moveTo(glm::vec3 position);
	void move(float forward, float direction, Camera camera, bool right, glm::vec3 position, float yaw);
	void draw(GLFWwindow* window);
	glm::vec3 getPosition();
private:
	GLuint vao;
	int indexCount;
	Shader shader;
	GLuint texture;
	glm::mat4 model_matrix;
	void updateVectors();
};