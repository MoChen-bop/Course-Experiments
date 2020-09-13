#pragma once
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "controller.h"

#include "shader.h"
#include "stb_image.h"

#include <vector>
using namespace std;

class Background
{
public:
	Background(float SIZE);
	static void change();
	void draw(GLFWwindow* window);
private:
	Shader shader;
	GLuint vao;
	GLuint texture[3];
	int indexCount;
};
