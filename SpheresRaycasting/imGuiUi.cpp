
#include "imGuiUi.hpp"

imGuiUi::imGuiUi(rendering& rendering) : m_rendering(rendering), io((ImGui::CreateContext(), ImGui::GetIO()))
{
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_rendering.window, true);
#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");
#endif
    ImGui_ImplOpenGL3_Init(m_rendering.glsl_version);
}

imGuiUi::~imGuiUi()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void imGuiUi::styleRounded()
{
    ImGuiStyle& style = ImGui::GetStyle();
    style.TabRounding = 8.f;
    style.FrameRounding = 8.f;
    style.GrabRounding = 8.f;
    style.WindowRounding = 8.f;
    style.PopupRounding = 8.f;
}

void imGuiUi::styleSquare()
{
    ImGuiStyle& style = ImGui::GetStyle();
    style.TabRounding = 0.f;
    style.FrameRounding = 0.f;
    style.GrabRounding = 0.f;
    style.WindowRounding = 0.f;
    style.PopupRounding = 0.f;
}

void imGuiUi::styleLight()
{
    ImGui::StyleColorsLight();
}

void imGuiUi::styleDark()
{
    ImGui::StyleColorsDark();
}

void imGuiUi::settingsWindow()
{
    ImGui::SetNextWindowSize(ImVec2(300, 80));
    ImGui::Begin("Dynamic settings", NULL, ImGuiWindowFlags_NoResize);

    ImGui::SetNextItemWidth(150);
    ImGui::ColorEdit3("Change background", (float*)&m_rendering.clear_color); // Edit 3 floats representing a color

    ImGui::Text("Average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::End();
}

void imGuiUi::newFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}
