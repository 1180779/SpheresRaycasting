
#ifndef U1180779_CAMERA_H
#define U1180779_CAMERA_H

#include "shader.hpp"
#include "rendering.hpp"
#include <glm/glm.hpp>

class camera 
{
public:
    /* sets window input mode to cursor disabled */
    static void disableCursor(const rendering& render);
    
    /* sets cursor callback */
    static void setCallbacks(const rendering& render);

    /* set mouse movement to modify this cameras yaw and pitch */
    void setCurrent();
    camera(const rendering& render, float speed = 2.f, float yaw = -90.f,
        float pitchMax = 89.f, float pitchMin = -89.f, 
        float fovMin = 1.0f, float fovMax = 45.0f)
        : m_render(render), m_cameraSpeed(speed), m_yaw(yaw),
        m_pitchMax(pitchMax), m_pitchMin(pitchMin), 
        m_fovMax(fovMax), m_fovMin(fovMin) 
    {
        m_fov = m_fovMax;
    }

    /* process input */
    camera& processInput();

    /* set 4f matrix with name in the shader */
    camera& use(shader& sh, const std::string& nameView = "view", const std::string& nameProj = "projection");

private:
    const rendering& m_render;
    glm::vec3 m_cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::vec3 m_cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 m_cameraUp = glm::vec3(0.f, 1.f, 0.f);
    glm::vec3 m_direction = glm::vec3(1.0f, 1.0f, 1.0f);

    friend void mouseCallback(GLFWwindow* window, double xpos, double ypos);
    friend void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    float m_cameraSpeed;
    float m_yaw;

    float m_pitch = 0.f;
    float m_pitchMax;
    float m_pitchMin;

    float m_fov;
    float m_fovMax;
    float m_fovMin;
};

static camera* currentCamera = nullptr;
static float lastX = 400, lastY = 300;
static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);


#endif