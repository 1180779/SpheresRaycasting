
/* CUDA */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "castRays.cuh"

/* NOT CUDA */
#include "rendering.hpp"
#include "imGuiUi.hpp"

#include "shader.hpp"
#include "shadersC.hpp"
#include "camera.hpp"

#include "camera.hpp"
#include "objectList.hpp"


#include "triangleShape.hpp"
#include "rectangleShape.hpp"
#include "cubeShape.hpp"



#define STB_IMAGE_IMPLEMENTATION

int main(int, char**)
{
    rendering render = rendering();
    imGuiUi ui = imGuiUi(render);
    ui.styleLight();
    ui.styleRounded();
    render.initGL();
    glEnable(GL_DEPTH_TEST);


    camera cam(render);
    cam.setCurrent();
    //camera::disableCursor(render);
    //camera::setCallbacks(render);


    // Main loop
    while (!glfwWindowShouldClose(render.window))
    {
        if (glfwGetKey(render.window, GLFW_KEY_ESCAPE)) {
            glfwSetWindowShouldClose(render.window, GL_TRUE);
        }
        render.measureDeltaTime();
        cam.processInput();
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();
        if (glfwGetWindowAttrib(render.window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        ui.newFrame();
        ui.settingsWindow();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(render.window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        render.clearColor();
        glClear(GL_DEPTH_BUFFER_BIT);


        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        render.swapBuffers();
    }
    return 0;
}