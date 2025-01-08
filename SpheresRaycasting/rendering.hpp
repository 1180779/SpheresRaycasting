
#ifndef U1180779_RENDERING_H
#define U1180779_RENDERING_H

#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "imGui/imgui.h"
#include <string>


class rendering 
{
public:
    rendering(std::string windowName = "openGL");
    ~rendering();

    rendering& measureDeltaTime();
    float deltaTime() const { return m_deltaTime; }

    void clearColor();
    void swapBuffers();
    void initGL();

    GLFWwindow* window;
    const char* glsl_version;
    ImVec4 clear_color = ImVec4(0.f, 0.f, 0.f, 1.00f);

private:
    float m_currFrame = 0.f;
    float m_lastFrame = 0.f;
    float m_deltaTime = 0.f;
};

#endif