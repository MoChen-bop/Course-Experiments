#pragma once
#include <iostream>
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"

class Camera
{
public:
	Camera(const glm::vec3& pos, float fov, float aspect, float zNear, float zFar, GLFWwindow* window);

	glm::mat4 GetViewProjection(GLFWwindow* window);

	~Camera() {}
private:
	glm::mat4 projection;
	glm::vec3 pos;
	glm::vec3 forward;
	glm::vec3 up;

	float speed = 3.0f;
	float mouseSpeed = 0.005f;
	glm::vec3 position = glm::vec3(0, 0, 5);
	float horizontalAngle = 3.14f;
	float verticalAngle = 0.0f;
	float initialFoV = 45.0f;
};