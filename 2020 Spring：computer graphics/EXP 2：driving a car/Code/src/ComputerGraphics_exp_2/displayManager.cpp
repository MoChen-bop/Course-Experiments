#include "displayManager.h"
#include "display.h"
#include "AntTweakBar/AntTweakBar.h"
#include "GLFW/glfw3.h"

void DisplayManager::createDisplay()
{
	display.setTitle("Computer Graphics Experiment 2: driving a car");
	display.setDisplayMode(DisplayMode(1024, 768));
	ContextAttri attri(4, 0);
	attri.withProfileCore(true);
	display.create(attri);

	GLFWwindow* window = display.getWindow();
	glfwSetMouseButtonCallback(window, (GLFWmousebuttonfun)TwEventMouseButtonGLFW);
	glfwSetCursorPosCallback(window, (GLFWcursorposfun)TwEventMousePosGLFW);
	glfwSetScrollCallback(window, (GLFWscrollfun)TwEventMouseWheelGLFW);
	glfwSetKeyCallback(window, (GLFWkeyfun)TwEventKeyGLFW);
	glfwSetCharCallback(window, (GLFWcharfun)TwEventCharGLFW);
}

void DisplayManager::clearDisplay()
{
	display.clear();
}

void DisplayManager::updateDisplay()
{
	display.update();
}

void DisplayManager::closeDisplay()
{
	display.destropy();
}

bool DisplayManager::isRequestClosed()
{
	return display.isRequestClosed();
}

bool DisplayManager::isKeyPressed(unsigned int keyCode)
{
	return display.isKeyPressed(keyCode);
}
