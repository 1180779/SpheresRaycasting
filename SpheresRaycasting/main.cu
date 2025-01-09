
/* CUDA */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "castRays.cuh"
#include "buffer.cuh"
#include "spheres.hpp"
#include "general.hpp"

/* NOT CUDA */
#include "rendering.hpp"
#include "imGuiUi.hpp"

#include "shader.hpp"
#include "camera.hpp"

int main(int, char**)
{
    rendering render = rendering();
    imGuiUi ui = imGuiUi(render);
    ui.styleLight();
    ui.styleRounded();
    render.initGL();
    glEnable(GL_DEPTH_TEST);

    
    xcudaSetDevice(0);



    buffer b = buffer();

    spheres data;
    data.generate(10, 50, 50, 0, 1920, 0, 1080, 50, 60);
    castRaysData raysData;
    raysData.sData = data.md_spheres;
   

    dim3 blocks = dim3(b.m_maxWidth / BLOCK_SIZE + 1, b.m_maxHeight / BLOCK_SIZE + 1);
    dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE);


    // Main loop
    while (!glfwWindowShouldClose(render.window))
    {
        if (glfwGetKey(render.window, GLFW_KEY_ESCAPE)) {
            glfwSetWindowShouldClose(render.window, GL_TRUE);
        }
        render.measureDeltaTime();
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

        b.mapCudaResource();


        raysData.width = b.m_maxWidth;
        raysData.height = b.m_maxHeight;
        raysData.surfaceObject = b.m_surfaceObject;
        castRaysKernel << <blocks, threads >> > (raysData);
        xcudaGetLastError();
        xcudaDeviceSynchronize();

        b.unmapCudaResource();
        b.use();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        render.swapBuffers();
    }
    data.free();
    return 0;
}