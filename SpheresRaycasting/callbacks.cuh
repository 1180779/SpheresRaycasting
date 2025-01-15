
#ifndef U1180779_CALLBACKS_CUH
#define U1180779_CALLBACKS_CUH

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include "cudaWrappers.hpp"
#include "transformScene.cuh"
#include "rendering.hpp"
#include "lbvhConcrete.cuh"
#include "lights.hpp"

static transformData* spheresDataForCallback = nullptr;
static const bvhDevice* spheresBvhForCallback = nullptr;
static lights* lightsCallback = nullptr;

static float lastX = 1280 / 2, lastY = 720 / 2;

void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void disableCursor(const rendering& render);

void disableCursor(const rendering& render)
{
    glfwSetInputMode(render.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
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

    const float sensitivity = 0.2f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;
    
    dim3 blocks = dim3(spheresDataForCallback->count / BLOCK_SIZE + 1);
    dim3 threads = dim3(BLOCK_SIZE);

    glm::mat4 t = glm::mat4(1.0f);
    t = glm::translate(t, glm::vec3(1920.0f / 2.0f, 1080.0f / 2.0f, 0.0f));
    t = glm::rotate(t, -glm::radians(xoffset), glm::vec3(0.0f, 1.0f, 0.0f));
    t = glm::rotate(t, glm::radians(yoffset), glm::vec3(1.0f, 0.0f, 0.0f));
    t = glm::translate(t, glm::vec3(-1920.0f / 2.0f, -1080.0f / 2.0f, 0.0f));
    spheresDataForCallback->t = t;

    transformSceneKernel<<<blocks, threads>>>(*spheresDataForCallback, *spheresBvhForCallback, *lightsCallback);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
}

#endif