#pragma once
#include "display.h"

class DisplayManager
{
public:
	DisplayManager() {}

	void createDisplay();
	void clearDisplay();
	void updateDisplay();
	void closeDisplay();

	GLFWwindow* getWindow() { return display.getWindow(); }

	bool isRequestClosed();
	bool isKeyPressed(unsigned int keyCode);

private:
	Display display;
};