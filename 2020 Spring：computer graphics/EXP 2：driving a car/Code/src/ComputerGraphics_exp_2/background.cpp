#include "background.h"
#include "loader.h"
#include "controller.h"

int texture_index = 0;

Background::Background(float SIZE)
{
    vector<std::string> image_set1
    {
        "../../resources/textures/skybox/right.jpg",
        "../../resources/textures/skybox/left.jpg",
        "../../resources/textures/skybox/top.jpg",
        "../../resources/textures/skybox/bottom.jpg",
        "../../resources/textures/skybox/front.jpg",
        "../../resources/textures/skybox/back.jpg",
    };

    vector<std::string> image_set2
    {
        "../../resources/textures/skybox_2/skybox_px.jpg",
        "../../resources/textures/skybox_2/skybox_nx.jpg",
        "../../resources/textures/skybox_2/skybox_py.jpg",
        "../../resources/textures/skybox_2/skybox_ny.jpg",
        "../../resources/textures/skybox_2/skybox_pz.jpg",
        "../../resources/textures/skybox_2/skybox_nz.jpg",
    };

    vector<std::string> image_set3
    {
        "../../resources/textures/skybox_3/nebula_px.jpg",
        "../../resources/textures/skybox_3/nebula_nx.jpg",
        "../../resources/textures/skybox_3/nebula_py.jpg",
        "../../resources/textures/skybox_3/nebula_ny.jpg",
        "../../resources/textures/skybox_3/nebula_pz.jpg",
        "../../resources/textures/skybox_3/nebula_nz.jpg",
    };

    vector<vector<std::string>> image_set = { image_set1, image_set2, image_set3 };

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

    std::vector<unsigned int> indices = {
        0,1,2, 2,3,0,
        1,5,6, 6,2,1,
        5,4,7, 7,6,5,
        4,0,3, 3,7,4,
        3,2,6, 6,7,3,
        4,5,1, 1,0,4
    };

    vao = Loader::getLoader()->loadVAO(vertices, indices);
    indexCount = indices.size();
    for (size_t i = 0; i < 3; i++) {
        texture[i] = Loader::getLoader()->loadCubemapTexture(image_set[i]);
    }
    texture_index = 0;
    shader = Shader("skybox.vs", "skybox.fs");
}

void Background::change()
{
    texture_index = (texture_index + 1) % 3;
}

void Background::draw(GLFWwindow* window)
{
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);
    shader.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture[texture_index]);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);

    loadMatrix(window, shader);

    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indexCount), GL_UNSIGNED_INT, (void*)0);

    glDisableVertexAttribArray(0);
    glBindVertexArray(0);
    glEnable(GL_CULL_FACE);
}