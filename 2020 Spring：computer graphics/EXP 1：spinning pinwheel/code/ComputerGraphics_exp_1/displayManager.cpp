#include "displayManager.h"
#include "display.h"

void DisplayManager::createDisplay()
{
	display.setTitle("Test");
	display.setDisplayMode(DisplayMode(1024, 768));
	ContextAttri attri(4, 0);
	display.create(attri);
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
	display.destroy();
}

bool DisplayManager::isRequestClosed()
{
	return display.isRequestClosed();
}

bool DisplayManager::isKeyPressed(unsigned int keyCode)
{
	return display.isKeyPressed(keyCode);
}
