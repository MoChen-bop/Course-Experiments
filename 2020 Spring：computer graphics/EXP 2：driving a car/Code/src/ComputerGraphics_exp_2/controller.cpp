#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "AntTweakBar/AntTweakBar.h"
#include "controller.h"
#include "background.h"
#include "ground.h"

#include <math.h>

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

Camera camera(glm::vec3(0.0f, 10.0f, 0.0f));

float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

float last_C_pressed = 0.0f;
float last_V_pressed = 0.0f;
float last_G_pressed = 0.0f;

int C_count = 0;
int V_count = 0;

float move_speed = 0;
float MAX_ACC_SPEED = 2000;
float MAX_SPEED = 20;
float F = 3;
float K = 1;
float OREN = 2;

bool control_car = true;

TwBar* bar;

void initBar()
{
    TwInit(TW_OPENGL_CORE, NULL);
    TwWindowSize(1024, 768);
    bar = TwNewBar("TweakBar");
    TwDefine(" GLOBAL help='Some settings about program parameters.' ");
    TwAddVarRO(bar, "speed", TW_TYPE_DOUBLE, &move_speed,
        " label='Car speed', help='Car speed (ralative speed)' ");
    TwAddVarRW(bar, "Position", TW_TYPE_QUAT4F, &camera.Position, "showval=true open=true ");
}

Camera getCamera()
{
    return camera;
}

bool is_control_car()
{
    return control_car;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    if (!control_car)
        camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (!control_car)
        camera.ProcessMouseScroll(yoffset);
}

void keyboard_event(GLFWwindow* window, Car* car)
{
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    float direction = 0;
    float deacc = K * move_speed;
    move_speed = move_speed - deacc * deltaTime;
    if (move_speed < 1)
        move_speed = 0;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        float acc = MAX_ACC_SPEED * sin(move_speed / MAX_SPEED * 3.14 / 2);
        if (acc == 0 && move_speed == 0)
            acc = MAX_ACC_SPEED / 10;
        move_speed = move_speed + acc * deltaTime;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {

    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        direction = - OREN * deltaTime * move_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        direction = OREN * deltaTime * move_speed;
    }


    if (control_car) {
        float move = move_speed * deltaTime;
        camera.changePosition(move, direction);
        car->run(move, direction, camera);
    }
    else {
        float move = 0;
        float direction = 0;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            move = 0.1;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            move = -0.1;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            direction = -0.1;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            direction = 0.1;
        }

        camera.changePosition(move, direction);
    }

    /*if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);*/

    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        if (last_C_pressed == 0) {
            Background::change();
            last_C_pressed = glfwGetTime();
        }
        else {
            if (glfwGetTime() - last_C_pressed > 0.2)
                Background::change();
            last_C_pressed = glfwGetTime();
        }
    }
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
        if (last_G_pressed == 0) {
            Ground::change();
            last_G_pressed = glfwGetTime();
        }
        else {
            if (glfwGetTime() - last_G_pressed > 0.2)
                Ground::change();
            last_G_pressed = glfwGetTime();
        }
    }
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) {
        control_car = false;
        if (last_V_pressed == 0) {
            camera.changeViewpoint();
            last_V_pressed = glfwGetTime();
        }
        else {
            if (glfwGetTime() - last_V_pressed > 0.2)
                camera.changeViewpoint();
            last_V_pressed = glfwGetTime();
        }
    }
        
}

void loadMatrix(GLFWwindow* window, Shader shader, glm::mat4 model)
{
    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
    shader.setMat4("projection", projection);

    glm::mat4 view = camera.GetViewMatrix();
    shader.setMat4("view", view);

    shader.setMat4("model", model);
}
