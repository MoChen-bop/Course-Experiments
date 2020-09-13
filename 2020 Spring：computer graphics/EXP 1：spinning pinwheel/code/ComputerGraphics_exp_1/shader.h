#pragma once

#include <string>
#include "GL/glew.h"
#include "transform.h"
#include "camera.h"

class Shader
{
public:
	Shader(const std::string& fileName);
	~Shader();

	void Bind();
	void Update(Transform& transform, Camera& camera, GLFWwindow* window);
private:
	static const unsigned int NUM_SHADERS = 2;
	enum {
		TRANSFORM_U,
		NUM_UNIFORM
	};
	GLuint program;
	GLuint shaders[NUM_SHADERS];
	GLuint uniforms[NUM_UNIFORM];
};