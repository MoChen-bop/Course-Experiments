#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
using namespace std;

#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "AntTweakBar/AntTweakBar.h"

#include "stb_image.h"
#include "displayManager.h"
#include "shader.h"
#include "camera.h"
#include "controller.h"
#include "model.h"
#include "background.h"
#include "ground.h"
#include "loader.h"
#include "car.h"
#include "light.h"
#include "grass.h"
#include "sign.h"


int main(int args, char* argv[])
{
    DisplayManager displayMgr;
    displayMgr.createDisplay();
    GLFWwindow* window = displayMgr.getWindow();
    initBar();
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glEnable(GL_DEPTH_TEST);

    Background background(200);
    Ground ground(200);
    Car car;
    std::vector<Grass> grass(60);
    std::vector<Sign> sign(30);
    for (int i = 0; i < 60; i++) {
        float size = 1.0 * rand() / RAND_MAX * 8;
        float x = 1.0 * rand() / RAND_MAX * 200 - 100;
        float z = 1.0 * rand() / RAND_MAX * 200 - 100;
        float yaw = 1.0 * rand() / RAND_MAX * 180 - 90;
        grass[i] = Grass(size, glm::vec3(x, 0.0f, z), yaw);
    }
    for (int i = 0; i < 20; i++) {
        float size = 1.0 * rand() / RAND_MAX * 4 + 4;
        float x = 1.0 * rand() / RAND_MAX * 200 - 100;
        float z = 1.0 * rand() / RAND_MAX * 200 - 100;
        float yaw = 1.0 * rand() / RAND_MAX * 180 - 90;
        sign[i] = Sign(size, glm::vec3(x, 0.0f, z), yaw);
    }

    cout << "'W': Run Forward." << endl;
    cout << "'A': Turn Left." << endl;
    cout << "'D': Turn Right." << endl;
    cout << "'C': Change Skybox." << endl;
    cout << "'G': Change Ground." << endl;
    cout << "'V': Switch Viewpoint." << endl;
    cout << "'Esc': Close Window." << endl;

    while (!displayMgr.isRequestClosed()) {
        displayMgr.clearDisplay();
        keyboard_event(window, &car);

        background.draw(window);
        ground.draw(window, getCamera(), car, is_control_car());
        car.draw(window);
        for (int i = 0; i < 60; i++)
            grass[i].draw(window);
        for (int i = 0; i < 20; i++)
            sign[i].draw(window);

        TwDraw();
        displayMgr.updateDisplay();
    }

    displayMgr.closeDisplay();
    return 0;
}
