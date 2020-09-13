#include "loader.h"
#include "stb_image.h"

#define VALS_PER_VERT 3
#define VALS_PER_NORMAL 3
#define VALS_PER_TEX 2

Loader* Loader::getLoader()
{
	return new Loader();
}

bool Loader::fileExists(const std::string& name)
{
	struct stat buffer {};
	return (stat(name.c_str(), &buffer) == 0);
}

GLuint Loader::loadVAO(const std::vector<float>& vertices, const std::vector<unsigned int>& indices)
{
	GLuint vaoHandle;
	glGenVertexArrays(1, &vaoHandle);
	glBindVertexArray(vaoHandle);

	unsigned int buffer[2];
	glGenBuffers(2, buffer);

	setupBuffer(buffer[0], vertices, 0, VALS_PER_VERT);
	setupIndicesBuffer(buffer[1], indices);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vaoHandle;
}

GLuint Loader::loadVAO(const std::vector<float>& vertices, const std::vector<unsigned int>& indices, const std::vector<float>& texCoords)
{
	GLuint vaoHandle;
	glGenVertexArrays(1, &vaoHandle);
	glBindVertexArray(vaoHandle);

	unsigned int buffer[3];
	glGenBuffers(3, buffer);

	setupBuffer(buffer[0], vertices, 0, VALS_PER_VERT);
	setupBuffer(buffer[1], texCoords, 1, VALS_PER_TEX);
	setupIndicesBuffer(buffer[2], indices);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vaoHandle;
}

GLuint Loader::loadVAO(const std::vector<float>& vertices, const std::vector<unsigned int>& indices, const std::vector<float>& texCoords, const std::vector<float>& normals)
{
	GLuint vaoHandle;
	glGenVertexArrays(1, &vaoHandle);
	glBindVertexArray(vaoHandle);

	unsigned int buffer[4];
	glGenBuffers(4, buffer);

	setupBuffer(buffer[0], vertices, 0, VALS_PER_VERT);
	setupBuffer(buffer[1], normals, 1, VALS_PER_NORMAL);
	setupBuffer(buffer[2], texCoords, 2, VALS_PER_TEX);
	setupIndicesBuffer(buffer[3], indices);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vaoHandle;
}

Image Loader::loadImage(const std::string& filepath)
{
	int x, y, n;
	unsigned char* data = stbi_load(filepath.c_str(), &x, &y, &n, 0);

	return Image(data, x, y, n);
}

GLuint Loader::loadCubemapTexture(const std::vector<std::string>& filenames)
{
	if (filenames.size() != 6) {
		std::cerr << "[Loader][Error] Cubemap requires 6 texture files." << std::endl;
		exit(1);
	}

	GLuint textureID;
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

	for (size_t i = 0; i < filenames.size(); i++) {
		std::cout << "[Loader] loading: " << filenames[i] << std::endl;
		if (!fileExists(filenames[i])) {
			std::cerr << "[Loader][Error] Skybox texture file " << i << " doesnt exist." << std::endl;
		}

		Image image = loadImage(filenames[i]);

		GLenum format = GL_RGB;
		if (image.channels == 4) {
			format = GL_RGBA;
		}

		glTexImage2D(static_cast<GLenum>(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i), 0, GL_RGB, image.width, image.height, 0,
			format, GL_UNSIGNED_BYTE, image.data);
		stbi_image_free(image.data);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	}

	return textureID;
}

GLuint Loader::loadTexture(const std::string& filepath)
{
	if (loadedTextures.count(filepath)) {
		std::cout << "[Loader] '" << filepath << "' already loaded, using cached texture." << std::endl;
		return loadedTextures[filepath];
	}

	std::cout << "[Loader] loading: " << filepath << std::endl;
	if (!fileExists(filepath)) {
		std::cerr << "[Loader] File doesnt exist, loading default texture." << std::endl;
		return loadDefaultTexture();
	}

	Image image = loadImage(filepath);

	GLuint textureID = loadTextureData(image.data, image.width, image.height, image.channels, GL_TEXTURE0);

	stbi_image_free(image.data);

	loadedTextures[filepath] = textureID;

	return textureID;
}

GLuint Loader::loadDefaultTexture()
{
	if (loadedTextures.count("DEFAULT_TEXTURE")) {
		std::cout << "[Loader] 'DEFAULT_TEXTURE' already loaded, using cached texture." << std::endl;
		return loadedTextures["DEFAULT_TEXTURE"];
	}

	const int SIZE = 64;
	GLubyte myimage[SIZE][SIZE][3];

	for (size_t i = 0; i < SIZE; i++) {
		for (size_t j = 0; j < SIZE; j++) {
			GLubyte c;
			c = (((i & 0x8) == 0) ^ ((j & 0x8) == 0)) * 255;
			myimage[i][j][0] = c;
			myimage[i][j][1] = c;
			myimage[i][j][2] = c;
		}
	}

	GLuint textureID = loadTextureData(&myimage[0][0][0], SIZE, SIZE, 3, GL_TEXTURE0);
	loadedTextures["DEFAULT_TEXTURE"] = textureID;

	return textureID;
}

GLuint Loader::loadTextureData(GLubyte* data, int x, int y, int n, GLenum textureUnit)
{
	GLuint textureID;

	glActiveTexture(textureUnit);
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	GLenum format = GL_RGB;

	if (n == 4) {
		format = GL_RGBA;
	}

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, x, y, 0, format, GL_UNSIGNED_BYTE, data);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);

	return textureID;
}

GLuint Loader::setupBuffer(unsigned int buffer, const std::vector<float>& values, int attributeIndex, int dataDimension)
{
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * values.size(), &values[0], GL_STATIC_DRAW);
	glVertexAttribPointer(attributeIndex, dataDimension, GL_FLOAT, GL_FALSE, 0, 0);

	return buffer;
}

GLuint Loader::setupIndicesBuffer(unsigned int buffer, const std::vector<unsigned int>& values)
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * values.size(), &values[0], GL_STATIC_DRAW);
	return buffer;
}

Image::Image() : data(nullptr), width(-1), height(-1), channels(-1) {}

Image::Image(unsigned char* data, int width, int height, int channels)
	: data(data), width(width), height(height), channels(channels) {
}

glm::vec3 Image::getPixel(int x, int y) const
{
	if (x < 0 || x >= width || y < 0 || y >= height) {
		return glm::vec3(-1.0f, -1.0f, -1.0f); 
	}
	int offset =
		((width * y) + x) * channels;  
	return glm::vec3((float)data[offset] / 255, (float)data[offset + 1] / 255, (float)data[offset + 2] / 255);
}
