#include <stdio.h>
#include <stdlib.h>

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "GL/glut.h"
#include "glm/glm.hpp"
using namespace glm;

#include "displayManager.h"
#include "mesh.h"
#include "transform.h"
#include "camera.h"
#include "shader.h"
#include "texture.h"
#include "controller.h"

int main(int argc, char* argv[])
{
	DisplayManager displayMgr;
	displayMgr.createDisplay();

	PinwheelMesh model;

	Camera camera(glm::vec3(0.0f, 0.0f, 5.0f), glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f, displayMgr.getWindow());
	Transform transform;
	Texture texture_bar("../../shader/texture_bar.jpg");
	Texture texture_fan("../../shader/texture_fan.jpg");
	Texture texture_screw("../../shader/texture_screw.jpg");
	Shader shader("../../shader/basicShader");
	Controller controller;

	while (!displayMgr.isRequestClosed()) {
		displayMgr.clearDisplay();
		model.spin(controller, displayMgr.getWindow());
		shader.Bind();
		shader.Update(transform, camera, displayMgr.getWindow());
		model.DrawFan();
		texture_fan.Bind();
		model.DrawBar();
		texture_bar.Bind();
		model.DrawScrew();
		texture_screw.Bind();
		displayMgr.updateDisplay();
	}

	displayMgr.closeDisplay();

	return 0;
}