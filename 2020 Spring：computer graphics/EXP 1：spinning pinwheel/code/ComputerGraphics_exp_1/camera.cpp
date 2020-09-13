#include "camera.h"

Camera::Camera(const glm::vec3& pos, float fov, float aspect, float zNear, float zFar, GLFWwindow* window)
{
	this->pos = pos;
	this->forward = glm::vec3(0.0f, 0.0f, 1.0f);
	this->up = glm::vec3(0.0f, 1.0f, 1.0f);
	this->projection = glm::perspective(fov, aspect, zNear, zFar);
	glfwSetCursorPos(window, 1024 / 2, 768 / 2);
}

glm::mat4 Camera::GetViewProjection(GLFWwindow* window)
{
	static double lastTime = glfwGetTime();

	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - lastTime);

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	glfwSetCursorPos(window, 1024 / 2, 768 / 2);

	horizontalAngle += mouseSpeed * float(1024 / 2 - xpos);
	verticalAngle += mouseSpeed * float(768 / 2 - ypos);

	glm::vec3 direction(
		cos(verticalAngle) * sin(horizontalAngle),
		sin(verticalAngle),
		cos(verticalAngle) * cos(horizontalAngle)
	);

	glm::vec3 right = glm::vec3(
		sin(horizontalAngle - 3.14 / 2.0f),
		0,
		cos(horizontalAngle - 3.14f / 2.0f)
	);

	glm::vec3 up = glm::cross(right, direction);

	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
		position += direction * deltaTime * speed;
	}

	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
		position -= direction * deltaTime * speed;
	}

	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		position += right * deltaTime * speed;
	}

	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
		position -= right * deltaTime * speed;
	}

	glm::mat4 viewMatrix = glm::lookAt(
		position,
		position + direction,
		up
	);

	lastTime = currentTime;

	return this->projection * viewMatrix;
}