#include "display.h"

#include <iostream>


bool Display::create(ContextAttri attr)
{
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW.\n");
		getchar();
		return false;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, attr.major);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, attr.minor);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	if (attr.bProfileCore) {
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	}

	window = glfwCreateWindow(displayMode.width, displayMode.height, title, nullptr, nullptr);
	if (!window) {
		fprintf(stderr, "Failed to create window.\n");
		getchar();
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);

	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW.\n");
		getchar();
		glfwTerminate();
		return false;
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetFramebufferSizeCallback(window, frameBufferSizeCallback);

	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	return true;
}

void Display::clear()
{
	glClear(GL_COLOR_BUFFER_BIT);
}

void Display::update()
{
	glfwPollEvents();
	glfwSwapBuffers(window);
	processEvent();
}

void Display::destroy()
{
	glfwDestroyWindow(window);
	glfwTerminate();
	window = nullptr;
}

bool Display::isRequestClosed()
{
	return glfwWindowShouldClose(window);
}

void Display::setDisplayMode(DisplayMode mode)
{
	displayMode = mode;
}

void Display::setTitle(const char* _title)
{
	title = _title;
}

void Display::frameBufferSizeCallback(GLFWwindow* _window, int w, int h)
{
	glViewport(0, 0, w, h);
}

bool Display::isKeyPressed(unsigned int keyCode)
{
	return glfwGetKey(window, keyCode) == GLFW_PRESS;
}

void Display::processEvent()
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
}
