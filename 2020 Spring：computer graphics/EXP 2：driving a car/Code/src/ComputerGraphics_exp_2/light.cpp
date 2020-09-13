#include "light.h"
#include <vector>
#include "loader.h"
#include "controller.h"

Light::Light()
{
    float SIZE = 0.3;
    std::vector<float> vertices = {
        -SIZE, -SIZE, SIZE,
        SIZE, -SIZE, SIZE,
        SIZE, SIZE, SIZE,
        -SIZE, SIZE, SIZE,
        -SIZE, -SIZE, -SIZE,
        SIZE, -SIZE, -SIZE,
        SIZE, SIZE, -SIZE,
        -SIZE, SIZE, -SIZE
    };

    std::vector<float> normals = {
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f,
        0.0f, 0.0f, -1.0f,
        0.0f, 0.0f, -1.0f,
        0.0f, 1.0f, 0.0f
    };

    std::vector<float> texCoords = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    std::vector<unsigned int> indices = {
        0,1,2, 2,3,0,
        1,5,6, 6,2,1,
        5,4,7, 7,6,5,
        4,0,3, 3,7,4,
        3,2,6, 6,7,3,
        4,5,1, 1,0,4
    };

    indexCount = indices.size();
    vao = Loader::getLoader()->loadVAO(vertices, indices, texCoords, normals);
	shader = Shader("light.vs", "light.fs");
    std::string texImage = "../../resources/textures/ground/arena.jpg";
    texture = Loader::getLoader()->loadTexture(texImage);
	position = glm::vec3(0.0f, 0.0f, 0.0f);
	front = glm::vec3(0.0f, 0.0f, -1.0f);
	Yaw = -90.0f;
	Pitch = 0.0f;
}

void Light::moveTo(glm::vec3 position)
{
    this->position = position;
    model_matrix = glm::translate(model_matrix, position);
}

void Light::move(float forward, float direction, Camera camera, bool right, glm::vec3 pos, float yaw)
{
    glm::vec3 go = forward * front;
    position = camera.Position;
    position.y -= 10;
    float _yaw = camera.Yaw;
    glm::mat4 temp;
    glm::mat4 rotate_1 = glm::rotate(temp, glm::radians(_yaw + 90), glm::vec3(0.0f, - 1.0f, 0.0f));
    glm::vec4 tran;
    if (right)
        tran = rotate_1 * glm::vec4(2.4f, 3.7f, -28.0f, 1.0f);
    else
        tran = rotate_1 * glm::vec4(-2.4f, 3.7f, -28.0f, 1.0f);
    glm::vec3 tr;
    tr.x = tran.x;
    tr.y = tran.y;
    tr.z = tran.z;
    model_matrix = glm::translate(temp, position + tr);

    position += go;
    Yaw += direction;
    updateVectors();
    /*position = pos;
    Yaw = yaw - 180;

    glm::mat4 temp;
    glm::mat4 rotate_1 = glm::rotate(temp, glm::radians(Yaw + 90), glm::vec3(0.0f, -1.0f, 0.0f));
    glm::vec4 tran;
    if (right)
        tran = rotate_1 * glm::vec4(2.4f, 3.7f, -28.0f, 1.0f);
    else
        tran = rotate_1 * glm::vec4(-2.4f, 3.7f, -28.0f, 1.0f);
    glm::vec3 tr;
    tr.x = tran.x;
    tr.y = tran.y;
    tr.z = tran.z;
    model_matrix = glm::translate(temp, position + tr);*/

    updateVectors();
}

void Light::draw(GLFWwindow* window)
{
    glDisable(GL_CULL_FACE);
    shader.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    loadMatrix(window, shader, model_matrix);

    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indexCount), GL_UNSIGNED_INT, (void*)0);

    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
    glEnable(GL_CULL_FACE);
}

glm::vec3 Light::getPosition()
{
    glm::vec4 temp(0.0f, 0.0f, 0.0f, 1.0f);
    temp = model_matrix * temp;
    position.x = temp.x;
    position.y = temp.y;
    position.z = temp.z;
	return position;
}

void Light::updateVectors()
{
	front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
	front.y = sin(glm::radians(Pitch));
	front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
	front = glm::normalize(front);
}
