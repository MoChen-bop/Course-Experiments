#include "display.h"

#include <iostream>
#include "AntTweakBar/AntTweakBar.h"

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
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
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
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetFramebufferSizeCallback(window, frameBufferSizeCallback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	//glEnable(GL_DEPTH_TEST);
	glfwSetCursorPos(window, 1024 / 2, 768 / 2);
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	return true;
}

void Display::clear()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Display::update()
{
	glfwSwapBuffers(window);
	glfwPollEvents();
	processEvent();
}

void Display::destropy()
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
	h = 1.0 * w / 1024 * 768;
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
