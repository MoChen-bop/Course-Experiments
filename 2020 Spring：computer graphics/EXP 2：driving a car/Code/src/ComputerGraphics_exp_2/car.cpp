#include "car.h"
#include "controller.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"


Car::Car()
{
	car_model_matrix = glm::rotate(car_model_matrix, glm::radians(90.0f), glm::vec3(0.0f, -1.0f, 0.0f));
	car_model_matrix = glm::rotate(car_model_matrix, glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f));
	car_model_matrix = glm::translate(car_model_matrix, glm::vec3(-20.0f, 0.0f, 3.65f));

	wheel_1_matrix = glm::rotate(wheel_1_matrix, glm::radians(90.f), glm::vec3(0.0f, 0.0f, -1.0f));
	wheel_1_matrix = glm::translate(wheel_1_matrix, glm::vec3(-1.65f, 3.0f, -26.0f));

	wheel_2_matrix = glm::rotate(wheel_2_matrix, glm::radians(90.f), glm::vec3(0.0f, 0.0f, -1.0f));
	wheel_2_matrix = glm::translate(wheel_2_matrix, glm::vec3(-1.65f, -3.0f, -26.0f));

	wheel_3_matrix = glm::rotate(wheel_3_matrix, glm::radians(90.f), glm::vec3(0.0f, 0.0f, -1.0f));
	wheel_3_matrix = glm::translate(wheel_3_matrix, glm::vec3(-1.65f, 3.0f, -15.0f));

	wheel_4_matrix = glm::rotate(wheel_4_matrix, glm::radians(90.f), glm::vec3(0.0f, 0.0f, -1.0f));
	wheel_4_matrix = glm::translate(wheel_4_matrix, glm::vec3(-1.65f, -3.0f, -15.0f));

	right_light.moveTo(glm::vec3(2.4f, 3.7f, -28.0f));
	left_light.moveTo(glm::vec3(-2.4f, 3.7f, -28.0f));

	car_model = Model("../../resources/objects/jeep.3DS");
	wheel_front_left = Model("../../resources/objects/wheel.3DS");
	wheel_front_right = Model("../../resources/objects/wheel.3DS");
	wheel_back_left = Model("../../resources/objects/wheel.3DS");
	wheel_back_right = Model("../../resources/objects/wheel.3DS");
	shader = Shader("car.vs", "car.fs");

	Position = glm::vec3(0.0f, 0.0f, 0.0f);
	Front = glm::vec3(0.0f, 0.0f, -1.0f);
	Yaw = 90.0f;
	Pitch = -90.0f;
}

void Car::draw(GLFWwindow* window)
{
	glDepthFunc(GL_LESS);
	shader.use();
	loadMatrix(window, shader, car_model_matrix);
	car_model.Draw(shader);
	loadMatrix(window, shader, wheel_1_matrix);
	wheel_front_left.Draw(shader);
	loadMatrix(window, shader, wheel_2_matrix);
	wheel_front_right.Draw(shader);
	loadMatrix(window, shader, wheel_3_matrix);
	wheel_back_left.Draw(shader);
	loadMatrix(window, shader, wheel_4_matrix);
	wheel_back_right.Draw(shader);
	right_light.draw(window);
	left_light.draw(window);
}

void Car::run(float forward, float direction, Camera camera)
{
	car_model_matrix = glm::translate(car_model_matrix, glm::vec3(20.0f, 0.0f, -3.65f));
	car_model_matrix = glm::rotate(car_model_matrix, glm::radians(direction), glm::vec3(0.0f, 0.0f, -1.0f));
	car_model_matrix = glm::translate(car_model_matrix, glm::vec3(-20.0f, 0.0f, 3.65f));

	wheel_1_matrix = glm::translate(wheel_1_matrix, glm::vec3(1.65f, -3.0f, 26.0f));
	wheel_1_matrix = glm::rotate(wheel_1_matrix, glm::radians(direction), glm::vec3(1.0f, 0.0f, 0.0f));
	wheel_1_matrix = glm::translate(wheel_1_matrix, glm::vec3(-1.65f, 3.0f, -26.0f));

	wheel_2_matrix = glm::translate(wheel_2_matrix, glm::vec3(1.65f, 3.0f, 26.0f));
	wheel_2_matrix = glm::rotate(wheel_2_matrix, glm::radians(direction), glm::vec3(1.0f, 0.0f, 0.0f));
	wheel_2_matrix = glm::translate(wheel_2_matrix, glm::vec3(-1.65f, -3.0f, -26.0f));

	wheel_3_matrix = glm::translate(wheel_3_matrix, glm::vec3(1.65f, -3.0f, 15.0f));
	wheel_3_matrix = glm::rotate(wheel_3_matrix, glm::radians(direction), glm::vec3(1.0f, 0.0f, 0.0f));
	wheel_3_matrix = glm::translate(wheel_3_matrix, glm::vec3(-1.65f, 3.0f, -15.0f));

	wheel_4_matrix = glm::translate(wheel_4_matrix, glm::vec3(1.65f, 3.0f, 15.0f));
	wheel_4_matrix = glm::rotate(wheel_4_matrix, glm::radians(direction), glm::vec3(1.0f, 0.0f, 0.0f));
	wheel_4_matrix = glm::translate(wheel_4_matrix, glm::vec3(-1.65f, -3.0f, -15.0f));

	glm::vec3 go = Front * forward;
	Position += go;
	Yaw += direction;

	glm::vec3 move(0.0f, 0.0f, 0.0f);
	move.x = go.z;
	move.y = go.x;
	move.z = go.y;

	car_model_matrix = glm::translate(car_model_matrix, move);
	
	move.x = go.y;
	move.y = go.x;
	move.z = go.z;
	wheel_1_matrix = glm::translate(wheel_1_matrix, move);
	wheel_2_matrix = glm::translate(wheel_2_matrix, move);
	wheel_3_matrix = glm::translate(wheel_3_matrix, move);
	wheel_4_matrix = glm::translate(wheel_4_matrix, move);

	updateVectors();

	right_light.move(forward, direction, camera, true, Position, Yaw);
	left_light.move(forward, direction, camera, false, Position, Yaw);
}

void Car::updateVectors()
{
	glm::vec3 front;
	front.y = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
	front.z = sin(glm::radians(Pitch));
	front.x = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
	Front = glm::normalize(front);
}
