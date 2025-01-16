
#include "imGuiUi.cuh"

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

void imGuiUi::constSettingsWindow(bool& start, materialGenerator::type& t)
{
    //ImGui::SetNextWindowSize(ImVec2(400, 150));
    ImGui::Begin("Constant settings", NULL, ImGuiWindowFlags_NoResize);
    
    ImGui::SetNextItemWidth(150);
    ImGui::ColorEdit3("Change background", (float*)&m_rendering.clear_color); // Edit 3 floats representing a color
    
    static int item = 0;
    if (ImGui::Combo("material type", &item, materialGenerator::typeString, 7))
    {
        t = static_cast<materialGenerator::type>(item);
    }
    

    start = ImGui::Button("Start");

    ImGui::Text("Average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::End();
}

void imGuiUi::settingsWindow(float& ia, bool& animate)
{
    ImGui::SetNextWindowSize(ImVec2(300, 150));
    ImGui::Begin("Dynamic settings", NULL, ImGuiWindowFlags_NoResize);

    ImGui::SetNextItemWidth(150);
    ImGui::ColorEdit3("Change background", (float*)&m_rendering.clear_color); // Edit 3 floats representing a color

    ImGui::Checkbox("Animate", &animate);

    ImGui::SetNextItemWidth(150);
    if(ImGui::InputFloat("Ia (ambient)", &ia, 0.005f, 0.01f)) 
    {
        if (ia < 0.0f)
            ia = 0.0f;
        if (ia > 1.0f)
            ia = 1.0f;
    }

    ImGui::Text("Average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::End();
}

void imGuiUi::checkInput()
{
    m_inputEscape = false;
    m_inputMouseClicked = false;
    m_inputMouseInView = true;
    if(ImGui::IsKeyPressed(ImGuiKey_Escape))
    {
        m_inputEscape = true;
    }
    if(ImGui::IsMouseClicked(ImGuiMouseButton_Left)) 
    {
        m_inputMouseClicked = true;
    }
}

void imGuiUi::handleInput()
{
    if(m_inputMouseLocked) 
    {
        if(m_inputEscape) 
        {
            /* release the mouse, reenable glfw mouse callback */
            glfwSetInputMode(m_rendering.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            glfwSetCursorPosCallback(m_rendering.window, ImGui_ImplGlfw_CursorPosCallback);
            m_inputMouseLocked = false;
        }
    }
    else 
    {
        if(m_inputMouseClicked && !io.WantCaptureMouse) 
        {
            /* capture the mouse */
            glfwSetInputMode(m_rendering.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            glfwSetCursorPosCallback(m_rendering.window, mouseCallbackRotateLights);
            m_inputMouseLocked = true;
        }
    }
}

void imGuiUi::newFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}
