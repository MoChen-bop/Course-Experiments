#pragma once
#include "camera.h"
#include "shader.h"
#include "car.h"
#include "GLFW/glfw3.h"

void initBar();
Camera getCamera();
bool is_control_car();
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void keyboard_event(GLFWwindow* window, Car* car);
void loadMatrix(GLFWwindow* window, Shader shader, glm::mat4 model = glm::mat4(1.0f));
