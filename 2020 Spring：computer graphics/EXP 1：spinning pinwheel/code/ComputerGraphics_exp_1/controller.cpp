#include "controller.h"
#include "GLFW/glfw3.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <iostream>


float Controller::omega = 0;
int Controller::direction = 1;
glm::mat4 Controller::spinMatrix;


Controller::Controller()
{
	spinMatrix = glm::mat4(1.0f);
}

glm::mat4& Controller::getMatrix()
{
	return spinMatrix;
}

void Controller::updateMatrix(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		if (alpha - drayFactor * omega * omega < 0) {
			omega = omega;
		}
		else {
			omega = omega + alpha - drayFactor * omega * omega;
		}
	}
	else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
		direction = -direction;
	}
	else {
		omega = omega - drayFactor * omega * omega * deltaTime;
	}
	if (omega < 0.001) omega = 0;
	//if (omega > 10) omega = 10;
	glm::mat4 temp = glm::mat4(1.0f);
	spinMatrix = glm::rotate(temp, glm::radians(direction * omega), glm::vec3(0.0f, 0.0f, 1.0f));
}

