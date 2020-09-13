#pragma once
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>

#include "GL/glew.h"
#include "GL/glut.h"


class ArgParser;
class Mesh;
class Camera;


class GLCanvas
{
public:
	static void initialize(ArgParser* _args, Mesh* _mesh);
private:
	static void InitLight();

	static ArgParser* args;
	static Mesh* mesh;
	static Camera* camera;

	static int mouseButton;
	static int mouseX;
	static int mouseY;
	static bool shiftPressed;
	static bool controlPressed;
	static bool altPressed;

	static void display(void);
	static void reshape(int w, int h);
	static void mouse(int button, int state, int x, int y);
	static void motion(int x, int y);
	static void keyboard(unsigned char key, int x, int y);
};

int HandleGLError(const std::string& message = "");