#include "Texture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "stb_image.h"

Texture::Texture(const std::string& fileName)
{
	data = stbi_load(fileName.c_str(), &width, &height, &numComponents, 4);

	if (data == nullptr) {
		std::cerr << "Can't load texture.\n" << std::endl;
		getchar();
	}

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

}

Texture::~Texture()
{
	glDeleteTextures(1, &texture);
	stbi_image_free(data);
}

void Texture::Bind()
{
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glBindTexture(GL_TEXTURE_2D, texture);
}
