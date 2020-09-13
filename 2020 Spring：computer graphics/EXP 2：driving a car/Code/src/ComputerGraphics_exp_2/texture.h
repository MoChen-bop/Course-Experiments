#pragma once
#include <string>
#include "GL/glew.h"

class Texture
{
public:
	Texture(const std::string& fileName);

	~Texture();

	void Bind();

private:
	GLuint texture;
	unsigned char* data;
	int width, height, numComponents;
};