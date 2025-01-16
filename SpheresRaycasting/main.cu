
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "lbvh/lbvh.cuh"

#include "imGuiUi.cuh"
#include "castRays.cuh"
#include "callbacks.cuh"

#include "rendering.hpp"
#include "buffer.hpp"
#include "cudaWrappers.hpp"
#include "shader.hpp"
#include "dataObject.hpp"
#include "timer.hpp"

int main(int, char**)
{
    /* init openGL and imGui (ui) */
    rendering render = rendering();
    imGuiUi ui = imGuiUi(render);
    ui.styleLight();
    ui.styleRounded();
    render.initGL();

    /* init CUDA */
    xcudaSetDevice(0);

    /* const settings loop */
    /*  ########################################################################## */
    materialGenerator::type matType;
    bool start = false;
    while (!glfwWindowShouldClose(render.window) && !start)
    {
        glfwPollEvents();
        if (glfwGetWindowAttrib(render.window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }
        ui.newFrame();
        ui.constSettingsWindow(start, matType);

        /* rendering */
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(render.window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        render.clearColor();
        glClear(GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        render.swapBuffers();
    }

    /* create openGL texture for CUDA */
    buffer b = buffer();

    /* generate data (spheres + lights) */
    timer t;
    t.start();
    dataObject data;
    data.generate(10000, range(50, 50), range(-1920, 1920), range(-1080, 1080), range(-4000, 4000), matType);
    data.generateLights(10, range(100, 100), range(-1920, 1920), range(-1080, 1080), range(-4000, 4000));
    t.stop("data.generate");

    /* lbvh (Linear Bounding Volume Hierarchy) */
    t.start();
    lbvh::bvh<float, unifiedObject, aabb_getter> bvh(data.m_objs.begin(), data.m_objs.end());
    t.stop("first tree generation");
    t.start();
    const auto ptrs = bvh.get_device_repr();
    t.stop("device repr");

    /* map data for callback functions (rotating objects with mouse) */
    transformData tData;
    tData.count = data.size();
    spheresDataForCallback = &tData;
    spheresBvhForCallback = &ptrs;
    lightsCallback = &data.md_lights;

    /* CUDA dimentions */
    dim3 blocks = dim3(b.m_maxWidth / BLOCK_SIZE + 1, b.m_maxHeight / BLOCK_SIZE + 1);
    dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE);

    /* main render loop */
    /*  ########################################################################## */

    while (!glfwWindowShouldClose(render.window))
    {
        glfwPollEvents();
        if (glfwGetWindowAttrib(render.window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }
        render.measureDeltaTime();
        ui.checkInput();

        ui.newFrame();
        ui.settingsWindow(data.md_lights.ia);

        /* rendering */
        ImGui::Render();
        ui.handleInput();
        int display_w, display_h;
        glfwGetFramebufferSize(render.window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        /* buffer clear is unnecessary, texture is drawn on whole screen */
        //render.clearColor();
        //glClear(GL_DEPTH_BUFFER_BIT);

        b.mapCudaResource();

        t.start();
        bvh.construct();
        t.stop("construct in loop");

        t.start();
        castRaysKernel<<<blocks, threads>>>(ptrs, b.m_maxWidth, b.m_maxHeight, b.m_surfaceObject, data.md_lights);
        xcudaDeviceSynchronize();
        xcudaGetLastError();
        t.stop("castRaysKernel");

        b.unmapCudaResource();
        b.use();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        render.swapBuffers();
    }
    data.freeLights();
    data.clear();
    return 0;
}