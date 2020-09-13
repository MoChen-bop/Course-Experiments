#include "shader.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
using namespace std;

#include <stdlib.h>
#include <string.h>

#include "GL/glew.h"

#include "camera.h"

static void CheckShaderError(GLuint shader, GLuint flag, bool isProgram, const std::string& errorMessage);
static std::string LoadShader(const std::string& fileName);
static GLuint CreateShader(const std::string& text, GLenum shaderType);

Shader::Shader(const std::string& fileName)
{
	program = glCreateProgram();

	shaders[0] = CreateShader(LoadShader(fileName + ".vs"), GL_VERTEX_SHADER);
	shaders[1] = CreateShader(LoadShader(fileName + ".fs"), GL_FRAGMENT_SHADER);

	for (unsigned int i = 0; i < NUM_SHADERS; i++) {
		glAttachShader(program, shaders[i]);
	}

	glBindAttribLocation(program, 0, "position");
	glBindAttribLocation(program, 1, "texCoord");

	glLinkProgram(program);
	CheckShaderError(program, GL_LINK_STATUS, true, "Error: Program Linking invalid");

	glValidateProgram(program);
	CheckShaderError(program, GL_VALIDATE_STATUS, true, "Error: Program Validate invalid");

	uniforms[TRANSFORM_U] = glGetUniformLocation(program, "transform");
}

Shader::~Shader()
{
	for (unsigned int i = 0; i < NUM_SHADERS; i++) {
		glDetachShader(program, shaders[i]);
		glDeleteShader(shaders[i]);
	}
	glDeleteProgram(program);
}

void Shader::Bind()
{
	glUseProgram(program);
}

void Shader::Update(Transform& transform, Camera& camera, GLFWwindow* window)
{
	glm::mat4 model = camera.GetViewProjection(window) * transform.GetModel();
	glUniformMatrix4fv(uniforms[TRANSFORM_U], 1, GL_FALSE, &model[0][0]);
}

void CheckShaderError(GLuint shader, GLuint flag, bool isProgram, const std::string& errorMessage)
{
	GLint success = 0;
	GLchar error[1024] = { 0 };

	if (isProgram) {
		glGetProgramiv(shader, flag, &success);
	}
	else {
		glGetShaderiv(shader, flag, &success);
	}

	if (success == GL_FALSE) {
		if (isProgram) {
			glGetProgramInfoLog(shader, sizeof(error), nullptr, error);
		}
		else {
			glGetShaderInfoLog(shader, sizeof(error), nullptr, error);
		}

		std::cerr << errorMessage << ":'" << error << "'" << std::endl;
	}
}

std::string LoadShader(const std::string& fileName)
{
	std::ifstream file;
	file.open((fileName).c_str());

	std::string output;
	std::string line;

	if (file.is_open()) {
		while (file.good()) {
			getline(file, line);
			output.append(line + "\n");
		}
	}
	else {
		std::cerr << "Unable to load shader: " << fileName << std::endl;
	}

	return output;
}

GLuint CreateShader(const std::string& text, GLenum shaderType)
{
	GLuint shader = glCreateShader(shaderType);

	if (shader == 0) {
		std::cerr << "Error: Shader creation failure" << std::endl;
	}

	const GLchar* shaderSourceString[1];
	GLint shaderSourceStringLength[1];
	shaderSourceString[0] = text.c_str();
	shaderSourceStringLength[0] = text.length();

	glShaderSource(shader, 1, shaderSourceString, shaderSourceStringLength);
	glCompileShader(shader);

	CheckShaderError(shader, GL_COMPILE_STATUS, false, "Error: Shader compilation failed");

	return shader;
}
