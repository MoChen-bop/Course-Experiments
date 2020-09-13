#include "camera.h"

glm::mat4 Camera::GetViewMatrix()
{
	return glm::lookAt(Position, Position + Front, Up);
}

void Camera::changeViewpoint()
{
	viewpoint = (viewpoint + 1) % 3;

	if (viewpoint == 0) {
		Position = glm::vec3(0.0f, 2.0f, 0.0f);
		Up = glm::vec3(0.0f, 1.0f, 0.0f);
		Front = glm::vec3(0.0f, 0.0f, -1.0f);
		Yaw = -90.0f;
		Pitch = 0;
	}
	else if (viewpoint == 1) {
		Position = glm::vec3(0.0f, 100.0f, 0.0f);
		Yaw = -90.0f;
		Pitch = -90.0f;
		updateCameraVectors();
	}
	else {
		Position = glm::vec3(0.0f, 5.0f, -50.0f);
		Yaw = 90;
		Pitch = 00;
	}
	updateCameraVectors();
}

void Camera::ProcessKeyboard(Camera_Movement direction, float deltaTime)
{
	float velocity = MovementSpeed * deltaTime;
	if (direction == FORWARD)
		Position += Front * velocity;
	if (direction == BACKWARD)
		Position -= Front * velocity;
	if (direction == LEFT)
		Position -= Right * velocity;
	if (direction == RIGHT)
		Position += Right * velocity;
}

void Camera::ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch)
{
	xoffset *= MouseSensitivity;
	yoffset *= MouseSensitivity;

	Yaw += xoffset;
	Pitch += yoffset;

	/*if (constrainPitch) {
		if (Pitch > 89.0f)
			Pitch = 89.0f;
		if (Pitch < -89.0f)
			Pitch = -89.0f;
	}*/

	updateCameraVectors();
}

void Camera::ProcessMouseScroll(float yoffset)
{
	if (Zoom >= 1.0f && Zoom <= 45.0f)
		Zoom -= yoffset;
	if (Zoom <= 1.0f)
		Zoom = 1.0f;
	if (Zoom >= 45.0f)
		Zoom = 45.0f;
}

void Camera::changePosition(float forward, float direction)
{
	Position += forward * Front;
	Yaw += direction;
	updateCameraVectors();
}

void Camera::updateCameraVectors()
{
	glm::vec3 front;
	front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
	front.y = sin(glm::radians(Pitch));
	front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
	Front = glm::normalize(front);

	Right = glm::normalize(glm::cross(Front, WorldUp));
	Up = glm::normalize(glm::cross(Right, Front));
}
