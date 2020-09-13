#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

#include "stb_image.h"
#include "mesh.h"
#include "shader.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>
using namespace std;

class Model
{
public:
	vector<TextureInfo> textures_loaded;
	vector<Mesh> meshed;
	string directory;
	bool gammaCorrection;

	Model() {}
	Model(string const& path, bool gamma = false) : gammaCorrection(gamma) { loadModel(path); }

	void Draw(Shader shader);

private:
	void loadModel(string const& path);

	void processNode(aiNode* node, const aiScene* scene);

	Mesh processMesh(aiMesh* mesh, const aiScene* scene);

	vector<TextureInfo> loadMaterialTextures(aiMaterial* mat, aiTextureType type, string typeName);
};