#pragma once
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "shader.h"
#include "model.h"
#include "light.h"
#include "camera.h"

class Car
{
public:
	Light right_light;
	Light left_light;

	Car();
	void draw(GLFWwindow* window);
	void run(float forward, float direction, Camera camera);
private:
	Shader shader;
	Model car_model;
	Model wheel_front_left;
	Model wheel_front_right;
	Model wheel_back_left;
	Model wheel_back_right;

	glm::mat4 car_model_matrix;

	glm::mat4 wheel_1_matrix;
	glm::mat4 wheel_2_matrix;
	glm::mat4 wheel_3_matrix;
	glm::mat4 wheel_4_matrix;

	glm::mat4 wheel_1_temp_matrix;
	glm::mat4 wheel_2_temp_matrix;
	glm::mat4 wheel_3_temp_matrix;
	glm::mat4 wheel_4_temp_matrix;

	glm::vec3 Position;
	glm::vec3 Front;
	float Yaw;
	float Pitch;

	void updateVectors();
};