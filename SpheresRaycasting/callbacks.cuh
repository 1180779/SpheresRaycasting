
#ifndef U1180779_CALLBACKS_H
#define U1180779_CALLBACKS_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include "general.hpp"
#include "transformScene.cuh"
#include "rendering.hpp"

static transformData* spheresDataForCallback = nullptr;
static float lastX = 1280 / 2, lastY = 720 / 2;

void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void disableCursor(const rendering& render);



void disableCursor(const rendering& render)
{
    glfwSetInputMode(render.window, GLFW_CURSOR, GLFW_CURSOR_CAPTURED);
    glfwSetCursorPosCallback(render.window, mouseCallback);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates range from bottom to top
    lastX = xpos;
    lastY = ypos;

    if (spheresDataForCallback == nullptr)
        return;

    dim3 blocks = dim3(spheresDataForCallback->sData.count / BLOCK_SIZE + 1);
    dim3 threads = dim3(BLOCK_SIZE);

    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    glm::mat4 t = glm::mat4(1.0f);
    t = glm::rotate(t, xoffset, glm::vec3(1.0f, 0.0f, 0.0f));
    spheresDataForCallback->t = glm::rotate(t, yoffset, glm::vec3(0.0f, 1.0f, 0.0f));

    transformSceneKernel<<<blocks, threads>>>(*spheresDataForCallback);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
}

#endif