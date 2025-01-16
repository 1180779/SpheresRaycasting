
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
    sceneConfig config;
    config.sCount = 1000;
    config.sXR = range(0, 1920);
    config.sYR = range(0, 1080);
    config.sZR = range(2000, 4000);
    config.sRR = range(50, 50);
    
    config.lCount = 10;
    config.lXR = range(0, 1920);
    config.lYR = range(0, 1080);
    config.lZR = range(2000, 4000);
    config.lRR = range(20, 20);

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
        ui.constSettingsWindow(start, config);

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

    /* preparation (data creation etc.) */
    /*  ########################################################################## */

    /* create openGL texture for CUDA */
    buffer b = buffer();

    /* generate data (spheres + lights) */
    timer t;
    t.start();
    dataObject data;
    data.generate(config.sCount, config.sRR, config.sXR, config.sYR, config.sZR, config.matType);
    data.generateLights(config.lCount, config.lRR, config.lXR, config.lYR, config.lZR);
    t.stop("data.generate");

    /* lbvh (Linear Bounding Volume Hierarchy) */
    t.start();
    lbvh::bvh<float, unifiedObject, aabb_getter> bvh(data.m_objs.begin(), data.m_objs.end());
    t.stop("first tree generation");
    t.start();
    const auto ptrs = bvh.get_device_repr();
    t.stop("device repr");

    /* map data for callback functions (rotating objects with mouse) */
    transformData tData, tDataAnimate;
    tData.count = data.size();
    tDataAnimate.count = data.size();
    spheresDataForCallback = &tData;
    spheresBvhForCallback = &ptrs;
    lightsCallback = &data.md_lights;
    shiftCallback = glm::vec3(config.sXR.avg(), config.sYR.avg(), config.sZR.avg());

    /* CUDA dimentions */
    dim3 blocks = dim3(b.m_maxWidth / BLOCK_SIZE + 1, b.m_maxHeight / BLOCK_SIZE + 1);
    dim3 threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksLinear = dim3(data.m_objs.size() / BLOCK_SIZE + 1);
    dim3 threadsLinear = dim3(BLOCK_SIZE);

    /* temp data for casting kernels */
    unifiedObjects objs;
    objs.dMalloc(b.m_maxWidth * b.m_maxHeight);

    /* main render loop */
    /*  ########################################################################## */

    bool animation = true;
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
        ui.settingsWindow(data.md_lights.ia, animation);

        /* rendering */
        ImGui::Render();
        ui.handleInput();
        int display_w, display_h;
        glfwGetFramebufferSize(render.window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        /* buffer clear is unnecessary, texture is drawn on whole screen */
        //render.clearColor();
        //glClear(GL_DEPTH_BUFFER_BIT);

        t.start();
        bvh.construct();
        t.stop("construct in loop");
        // animate (rotate with time)

        if(animation) 
        {
            float deltaTime = render.getDeltaTime();
            constexpr float sensitivityX = 50.0f;
            constexpr float sensitivityY = 0.0f;
            float xoffset = deltaTime * sensitivityX;
            float yoffset = deltaTime * sensitivityY;

            glm::mat4 t = glm::mat4(1.0f);
            t = glm::translate(t, shiftCallback);
            t = glm::rotate(t, -glm::radians(xoffset), glm::vec3(0.0f, 1.0f, 0.0f));
            t = glm::rotate(t, glm::radians(yoffset), glm::vec3(1.0f, 0.0f, 0.0f));
            t = glm::translate(t, -shiftCallback);
            tDataAnimate.t = t;
            callbackLightsKernel<<<blocksLinear, threadsLinear>>>(tDataAnimate, ptrs, data.md_lights);
            xcudaDeviceSynchronize();
            xcudaGetLastError();
            // TODO: add animation
        }

        b.mapCudaResource();

        /* cast rays */
        t.start();
        data.md_lights.clearColor.x = render.clear_color.x; /* copy background color data */
        data.md_lights.clearColor.y = render.clear_color.y;
        data.md_lights.clearColor.z = render.clear_color.z;
        data.md_lights.clearColor.w = render.clear_color.w;
        findClosestKernel<<<blocks, threads>>>(ptrs, b.m_maxWidth, b.m_maxHeight, objs, data.md_lights);
        xcudaDeviceSynchronize();
        xcudaGetLastError();
        t.stop("findClosestKernel");

        t.start();
        drawColorKernel<<<blocks, threads>>>(objs, b.m_maxWidth, b.m_maxHeight, b.m_surfaceObject, data.md_lights);
        xcudaDeviceSynchronize();
        xcudaGetLastError();
        t.stop("drawColorKernel");

        b.unmapCudaResource();
        b.use();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        render.swapBuffers();
    }
    objs.dFree();
    data.freeLights();
    data.clear();
    return 0;
}