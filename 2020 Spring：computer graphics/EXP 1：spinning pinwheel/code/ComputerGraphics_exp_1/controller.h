#pragma once

#include "GLFW/glfw3.h"
#include "glm/glm.hpp"

class Controller
{
public:
	Controller();
	glm::mat4& getMatrix();
	void updateMatrix(GLFWwindow* window);

private:
	static glm::mat4 spinMatrix;

	int menu;

	float alpha = 100;
	static float omega;
	float drayFactor = 0.1;
	float deltaTime = 0.01;
	static int direction;

};