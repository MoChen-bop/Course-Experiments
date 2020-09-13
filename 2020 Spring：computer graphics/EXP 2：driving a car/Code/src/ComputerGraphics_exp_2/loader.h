#pragma once
#include <sys/stat.h>
#include "glad/glad.h"
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "glm/glm.hpp"

struct Image {
    unsigned char* data;
    int width;
    int height;
    int channels;

    Image();
    Image(unsigned char* data, int width, int height, int channels);
    glm::vec3 getPixel(int x, int y) const;
};

class Loader
{
public:
	static Loader* getLoader();

    static bool fileExists(const std::string& name);

    static GLuint loadVAO(const std::vector<float>& vertices, const std::vector<unsigned int>& indices);
    static GLuint loadVAO(const std::vector<float>& vertices, const std::vector<unsigned int>& indices,
        const std::vector<float>& texCoords);
    static GLuint loadVAO(const std::vector<float>& vertices, const std::vector<unsigned int>& indices,
        const std::vector<float>& texCoords, const std::vector<float>& normals);

    static Image loadImage(const std::string& filepath);
    static GLuint loadCubemapTexture(const std::vector<std::string>& filenames);
    GLuint loadTexture(const std::string& filepath);
    GLuint loadDefaultTexture();

private:
    //static Loader* loader;
	//Loader() = default;
    std::map<std::string, GLuint> loadedTextures;

    static GLuint loadTextureData(GLubyte* data, int x, int y, int n, GLenum textureUnit);
    static GLuint setupBuffer(unsigned int buffer, const std::vector<float>& values, int attributeIndex, int dataDimension);
    static GLuint setupIndicesBuffer(unsigned int buffer, const std::vector<unsigned int>& values);

};