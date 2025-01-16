
#include "callbacks.cuh"

transformData* spheresDataForCallback = nullptr;
const bvhDevice* spheresBvhForCallback = nullptr;
lights* lightsCallback = nullptr;

glm::vec3 shiftCallback = glm::vec3(1920.0f / 2.0f, 1080.0f / 2.0f, 0.0f);


void disableCursor(const rendering& render)
{
    glfwSetInputMode(render.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(render.window, mouseCallbackRotateLights);
}

void mouseCallbackRotateAll(GLFWwindow* window, double xpos, double ypos)
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

    dim3 blocks = dim3(spheresDataForCallback->count / BLOCK_SIZE1D + 1);
    dim3 threads = dim3(BLOCK_SIZE1D);

    glm::mat4 t = glm::mat4(1.0f);
    t = glm::translate(t, shiftCallback);
    t = glm::rotate(t, -glm::radians(xoffset), glm::vec3(0.0f, 1.0f, 0.0f));
    t = glm::rotate(t, glm::radians(yoffset), glm::vec3(1.0f, 0.0f, 0.0f));
    t = glm::translate(t, -shiftCallback);
    spheresDataForCallback->t = t;

    callbackAllKernel<<<blocks, threads>>>(*spheresDataForCallback, *spheresBvhForCallback, *lightsCallback);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
}

void mouseCallbackRotateLights(GLFWwindow* window, double xpos, double ypos)
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

    dim3 blocks = dim3(spheresDataForCallback->count / BLOCK_SIZE1D + 1);
    dim3 threads = dim3(BLOCK_SIZE1D);

    glm::mat4 t = glm::mat4(1.0f);
    t = glm::translate(t, glm::vec3(1920.0f / 2.0f, 1080.0f / 2.0f, 0.0f));
    t = glm::rotate(t, -glm::radians(xoffset), glm::vec3(0.0f, 1.0f, 0.0f));
    t = glm::rotate(t, glm::radians(yoffset), glm::vec3(1.0f, 0.0f, 0.0f));
    t = glm::translate(t, glm::vec3(-1920.0f / 2.0f, -1080.0f / 2.0f, 0.0f));
    spheresDataForCallback->t = t;

    callbackLightsKernel<<<blocks, threads>>>(*spheresDataForCallback, *spheresBvhForCallback, *lightsCallback);
    xcudaDeviceSynchronize();
    xcudaGetLastError();
}
