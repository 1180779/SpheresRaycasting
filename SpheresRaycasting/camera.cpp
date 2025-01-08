
#include "camera.hpp"

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates range from bottom to top
    lastX = xpos;
    lastY = ypos;

    if (!currentCamera)
        return;

    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    currentCamera->m_yaw += xoffset;
    currentCamera->m_pitch += yoffset;
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (!currentCamera)
        return;
    currentCamera->m_fov -= (float)yoffset;
    if (currentCamera->m_fov < currentCamera->m_fovMin)
        currentCamera->m_fov = currentCamera->m_fovMin;
    if (currentCamera->m_fov > currentCamera->m_fovMax)
        currentCamera->m_fov = currentCamera->m_fovMax;
}



void camera::disableCursor(const rendering& render)
{
    glfwSetInputMode(render.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void camera::setCallbacks(const rendering& render)
{
    glfwSetCursorPosCallback(render.window, mouseCallback);
    glfwSetScrollCallback(render.window, scrollCallback);
}

void camera::setCurrent()
{
    currentCamera = this;
}

camera& camera::processInput()
{
    float speed = m_cameraSpeed * m_render.deltaTime();
    if (glfwGetKey(m_render.window, GLFW_KEY_W) == GLFW_PRESS) {
        m_cameraPos += speed * m_cameraFront;
    }
    if (glfwGetKey(m_render.window, GLFW_KEY_S) == GLFW_PRESS) {
        m_cameraPos -= speed * m_cameraFront;
    }
    if (glfwGetKey(m_render.window, GLFW_KEY_A) == GLFW_PRESS) {
        m_cameraPos -= speed * glm::normalize(glm::cross(m_cameraFront, m_cameraUp));
    }
    if (glfwGetKey(m_render.window, GLFW_KEY_D) == GLFW_PRESS) {
        m_cameraPos += speed * glm::normalize(glm::cross(m_cameraFront, m_cameraUp));
    }
    return* this;
}

camera& camera::use(shader& sh, const std::string& nameView, const std::string& nameProj)
{
    m_direction.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    m_direction.y = sin(glm::radians(m_pitch));
    m_direction.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    m_cameraFront = glm::normalize(m_direction);

    sh.setMatrix4fv(nameProj, glm::perspective(glm::radians(m_fov), 800.0f / 600.0f, 0.1f, 100.0f));
    sh.setMatrix4fv(nameView, glm::lookAt(m_cameraPos, m_cameraPos + m_cameraFront, m_cameraUp));
    return *this;
}
