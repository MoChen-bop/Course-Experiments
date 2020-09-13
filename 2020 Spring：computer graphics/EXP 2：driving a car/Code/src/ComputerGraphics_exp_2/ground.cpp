#include "ground.h"

int ground_texture_index = 0;

Ground::Ground(float SIZE)
{
	std::vector<float> vertices(900 * 4 * 3);
	std::vector<unsigned int> indices(900 * 3 * 2);
	std::vector<float> texCoords(900 * 4 * 2);
	std::vector<float> normals(900 * 4 * 3);
	int index = 0;
	for (int x = -150; x < 150; x += 10) {
		for (int z = -150; z < 150; z += 10) {
			vertices[index * 4 * 3] = x;
			vertices[index * 4 * 3 + 1] = 0;
			vertices[index * 4 * 3 + 2] = z;
			normals[index * 4 * 3] = 0;
			normals[index * 4 * 3 + 1] = 1;
			normals[index * 4 * 3 + 2] = 0;

			vertices[index * 4 * 3 + 3] = x + 10;
			vertices[index * 4 * 3 + 3 + 1] = 0;
			vertices[index * 4 * 3 + 3 + 2] = z;
			normals[index * 4 * 3 + 3] = 0;
			normals[index * 4 * 3 + 3 + 1] = 1;
			normals[index * 4 * 3 + 3 + 2] = 0;

			vertices[index * 4 * 3 + 6] = x + 10;
			vertices[index * 4 * 3 + 6 + 1] = 0;
			vertices[index * 4 * 3 + 6 + 2] = z + 10;
			normals[index * 4 * 3 + 6] = 0;
			normals[index * 4 * 3 + 6 + 1] = 1;
			normals[index * 4 * 3 + 6 + 2] = 0;

			vertices[index * 4 * 3 + 9] = x;
			vertices[index * 4 * 3 + 9 + 1] = 0;
			vertices[index * 4 * 3 + 9 + 2] = z + 10;
			normals[index * 4 * 3 + 9] = 0;
			normals[index * 4 * 3 + 9 + 1] = 1;
			normals[index * 4 * 3 + 9 + 2] = 0;

			texCoords[index * 4 * 2] = x / 10;
			texCoords[index * 4 * 2 + 1] = z / 10;

			texCoords[index * 4 * 2 + 2] = (x + 10) / 10;
			texCoords[index * 4 * 2 + 3] = z / 10;

			texCoords[index * 4 * 2 + 4] = (x + 10) / 10;
			texCoords[index * 4 * 2 + 5] = (z + 10) / 10;

			texCoords[index * 4 * 2 + 6] = x / 10;
			texCoords[index * 4 * 2 + 7] = (z + 10) / 10;

			indices[index * 3 * 2] = index * 4;
			indices[index * 3 * 2 + 1] = index * 4 + 1;
			indices[index * 3 * 2 + 2] = index * 4 + 3;

			indices[index * 3 * 2 + 3] = index * 4 + 1;
			indices[index * 3 * 2 + 4] = index * 4 + 2;
			indices[index * 3 * 2 + 5] = index * 4 + 3;

			index += 1;
		}
	}

	vao = Loader::getLoader()->loadVAO(vertices, indices, texCoords, normals);
	indexCount = indices.size();
	std::vector<std::string> texImage = {
		"../../resources/textures/ground/arena.jpg",
		"../../resources/textures/ground/Asphalt_col.jpg",
		"../../resources/textures/ground/sand0.BMP"
	};

	for (size_t i = 0; i < 3; i++) {
		texture[i] = Loader::getLoader()->loadTexture(texImage[i]);
	}
	ground_texture_index = 0;
	shader = Shader("ground.vs", "ground.fs");

	shader.setInt("material.diffuse", 0);
	shader.setInt("material.specular", 1);
}

void Ground::change()
{
	ground_texture_index = (ground_texture_index + 1) % 3;
}

void Ground::draw(GLFWwindow* window, Camera camera, Car car, bool control_car)
{
	glDisable(GL_CULL_FACE);
	shader.use();
	shader.setVec3("viewPos", camera.Position);
	shader.setFloat("material.shininess", 32.0f);

	shader.setVec3("dirLight.direction", -0.2f, -1.0f, -0.3f);
	shader.setVec3("dirLight.ambient", 0.05f, 0.05f, 0.05f);
	shader.setVec3("dirLight.diffuse", 0.4f, 0.4f, 0.4f);
	shader.setVec3("dirLight.specular", 0.5f, 0.5f, 0.5f);

	if (control_car) {
		position = camera.Position;
		position.y -= 10;
	}
	if (control_car) {
		front = camera.Front;
		yaw = camera.Yaw;
	}
	glm::mat4 temp;
	glm::mat4 rotate = glm::rotate(temp, glm::radians(yaw + 90), glm::vec3(0.0f, -1.0f, 0.0f));
	glm::vec4 tran_1;
	glm::vec4 tran_2;
	tran_1 = rotate * glm::vec4(2.4f, 3.7f, -28.0f, 1.0f);
	tran_2 = rotate * glm::vec4(-2.4f, 3.7f, -28.0f, 1.0f);
	glm::vec3 p_1 = position + glm::vec3(tran_1);
	glm::vec3 p_2 = position + glm::vec3(tran_2);

	shader.setVec3("spotLight[1].position", p_2);
	shader.setVec3("spotLight[1].direction", front + glm::vec3(0.0f, -0.0f, 0.0f));
	shader.setVec3("spotLight[1].ambient", 0.0f, 0.0f, 0.0f);
	shader.setVec3("spotLight[1].diffuse", 1.0f, 1.0f, 1.0f);
	shader.setVec3("spotLight[1].specular", 1.0f, 1.0f, 1.0f);
	shader.setFloat("spotLight[1].constant", 1.0f);
	shader.setFloat("spotLight[1].linear", 0.09);
	shader.setFloat("spotLight[1].quadratic", 0.032);
	shader.setFloat("spotLight[1].cutOff", glm::cos(glm::radians(60.5f)));
	shader.setFloat("spotLight[1].outerCutOff", glm::cos(glm::radians(90.0f)));

	shader.setVec3("spotLight[0].position", p_1);
	shader.setVec3("spotLight[0].direction", front + glm::vec3(0.0f, -0.0f, 0.0f));
	shader.setVec3("spotLight[0].ambient", 0.0f, 0.0f, 0.0f);
	shader.setVec3("spotLight[0].diffuse", 1.0f, 1.0f, 1.0f);
	shader.setVec3("spotLight[0].specular", 1.0f, 1.0f, 1.0f);
	shader.setFloat("spotLight[0].constant", 1.0f);
	shader.setFloat("spotLight[0].linear", 0.0009);
	shader.setFloat("spotLight[0].quadratic", 0.00032);
	shader.setFloat("spotLight[0].cutOff", glm::cos(glm::radians(30.5f)));
	shader.setFloat("spotLight[0].outerCutOff", glm::cos(glm::radians(90.0f)));

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture[ground_texture_index]);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, texture[ground_texture_index]);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	loadMatrix(window, shader);

	glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indexCount), GL_UNSIGNED_INT, (void*)0);

	glDisableVertexAttribArray(0);
	glBindVertexArray(0);
	glEnable(GL_CULL_FACE);
}
