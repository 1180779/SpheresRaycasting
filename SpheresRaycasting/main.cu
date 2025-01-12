
/* CUDA */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "castRays.cuh"
#include "buffer.cuh"
#include "general.hpp"

#include "callbacks.cuh"

/* NOT CUDA */
#include "rendering.hpp"
#include "imGuiUi.hpp"

#include "shader.hpp"

#include "mat4.cuh"

#include "dataObject.hpp"

void matTests() {
    glm::mat4 tGLM = glm::rotate(glm::mat4(1.0f), 180.0f, glm::vec3(1.0f, 0.0f, 0.0f));
    mat4 t;
    t = tGLM;

    glm::vec4 vGLM(1.f, 1.f, 1.f, 1.f);
    vec4 v(1.f, 1.f, 1.f, 1.f);

    v = t * v;
    vGLM = tGLM * vGLM;

    std::cout << "x = " << v(0) << ", y = " << v(1) << ", z = " << v(2) << ", wGLM = " << v(3) << std::endl;
    std::cout << "xGLM = " << vGLM[0] << ", yGLM = " << vGLM[1] << ", zGLM = " << vGLM[2] << ", wGLM = " << vGLM[3] << std::endl;
}

int main(int, char**)
{
    //matTests();
    //return;

    rendering render = rendering();
    imGuiUi ui = imGuiUi(render);
    ui.styleLight();
    ui.styleRounded();
    render.initGL();

    xcudaSetDevice(0);

    disableCursor(render);

    buffer b = buffer();


    dataObject data;
    data.generate(200, 50, 50, -1920, 1920, -1080, 1080, 100, 200);

    castRaysData raysData;
    raysData.data = data.md_unified;
   
    transformData tData;
    tData.data = data.md_unified;

    spheresDataForCallback = &tData;

    dim3 blocksForSpheres = dim3(data.dCount() / BLOCK_SIZE + 1);
    dim3 threadsForSpheres = dim3(BLOCK_SIZE);

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
        xcudaDeviceSynchronize();
        xcudaGetLastError();

        //data.mh_spheres.copyDeviceToHost(raysData.sData);
        //std::cout << "\n\n";
        //std::cout << "DATA" << std::endl;
        //for (int i = 0; i < data.mh_spheres.count; ++i) {
        //    std::cout << "x = " << data.mh_spheres.x[i] << ", y = " << data.mh_spheres.y[i] << ", z = " << data.mh_spheres.z[i] << ", r = " << data.mh_spheres.r[i] << std::endl;
        //}

        b.unmapCudaResource();
        b.use();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        render.swapBuffers();
    }
    data.free();
    return 0;
}