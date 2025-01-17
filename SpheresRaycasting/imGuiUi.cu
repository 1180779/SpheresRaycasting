
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

static void rangeControl(const char* nameMin, const char* nameMax, range& r)
{
    ImGui::SetNextItemWidth(150);
    if (ImGui::InputFloat(nameMin, &r.min, 10.0f, 100.0f))
    {
        if (r.min > r.max)
            r.min = r.max;
    }
    ImGui::SetNextItemWidth(150);
    if (ImGui::InputFloat(nameMax, &r.max, 10.0f, 100.0f))
    {
        if (r.max < r.min)
            r.max = r.min;
    }
}

static void rangeControlWithLimits(const char* nameMin, const char* nameMax, range& r, range limits)
{
    ImGui::SetNextItemWidth(150);
    if (ImGui::InputFloat(nameMin, &r.min, 0.05f, 0.2f))
    {
        if (r.min > r.max)
            r.min = r.max;
        if (r.min < limits.min)
            r.min = limits.min;
    }
    ImGui::SetNextItemWidth(150);
    if (ImGui::InputFloat(nameMax, &r.max, 0.05f, 0.2f))
    {
        if (r.max < r.min)
            r.max = r.min;
        if (r.max > limits.max)
            r.max = limits.max;
    }
}

static void countControl(const char* name, unsigned int& c) 
{
    ImGui::SetNextItemWidth(150);
    if (ImGui::InputInt(name, (int*)&c, 1, 10))
    {
        if ((int)c < 0)
            c = 0;
    }
}
void imGuiUi::constSettingsWindow(bool& start, sceneConfig& config, int& camWidth, int& camHeight)
{
    /* start window */
    constexpr float spacing = 15.0f;
    ImGuiWindowFlags winFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

    /* sphere settings window */
    ImGui::SetNextWindowPos(ImVec2(20, 20));
    ImGui::SetNextWindowSize(ImVec2(400, 620));
    ImGui::Begin("Sphere settings", NULL, winFlags);
    
    ImGui::Dummy(ImVec2(0.0f, spacing));
    countControl("spheres count", config.sCount);
    rangeControl("spheres x range min", "spheres x range max", config.sXR);
    rangeControl("spheres y range min", "spheres y range max", config.sYR);
    rangeControl("spheres z range min", "spheres z range max", config.sZR);
    rangeControl("spheres radius range min", "spheres radius range max", config.sRR);

    ImGui::Dummy(ImVec2(0.0f, spacing));
    ImGui::Combo("material type", (int*)(&config.matType), 
        materialGenerator::typeString, materialGenerator::typeCount);
    ImGui::End();

    /* light settings window */
    ImGui::SetNextWindowPos(ImVec2(440, 20));
    ImGui::SetNextWindowSize(ImVec2(400, 620));
    ImGui::Begin("Light settings", NULL, winFlags);
    ImGui::Dummy(ImVec2(0.0f, spacing));
    countControl("lights count", config.lCount);
    rangeControl("lights x range min", "lights x range max", config.lXR);
    rangeControl("lights y range min", "lights y range max", config.lYR);
    rangeControl("lights z range min", "lights z range max", config.lZR);
    rangeControl("lights radius range min", "lights radius range max", config.lRR);

    rangeControlWithLimits("light Is min", "light Is max", config.isR, range(0.0f, 1.0f));
    rangeControlWithLimits("light Id min", "light Id max", config.idR, range(0.0f, 1.0f));
    ImGui::End();


    /* other settings and start window */
    ImGui::SetNextWindowPos(ImVec2(860, 20));
    ImGui::SetNextWindowSize(ImVec2(400, 200));
    ImGui::Begin("Settings", NULL, winFlags);
    ImGui::SetNextItemWidth(150);
    ImGui::ColorEdit3("Change background", (float*)&m_rendering.clear_color);

    ImGui::Text("Camera resolution");
    ImGui::SetNextItemWidth(150);
    if (ImGui::InputInt("camera res x", (int*)&camWidth, 10, 100))
    {
        if (camWidth > 1920)
            camWidth = 1920;
        if (camWidth < 1280)
            camWidth = 1280;
    }
    ImGui::SetNextItemWidth(150);
    if (ImGui::InputInt("camera res y", (int*)&camHeight, 10, 100))
    {
        if (camHeight > 1080)
            camHeight = 1080;
        if (camHeight < 720)
            camHeight = 720;
    }

    ImGui::Dummy(ImVec2(0.0f, spacing));
    start = ImGui::Button("Start");
    if(ImGui::Button("Load from file")) 
    {
        config.loadFromFile("config.txt");
    }

    ImGui::Text("Average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::End();
}

void imGuiUi::settingsWindow(float& ia, bool& animate)
{
    ImGui::SetNextWindowSize(ImVec2(300, 200));
    ImGui::Begin("Dynamic settings", NULL, ImGuiWindowFlags_NoResize);

    ImGui::SetNextItemWidth(150);
    ImGui::ColorEdit3("Change background", (float*)&m_rendering.clear_color);

    ImGui::Checkbox("Animate", &animate);

    ImGui::BeginGroup();
    ImGui::Text("Rotation mode");
    if(ImGui::RadioButton("spheres", m_rotateObjects))
    {
        m_rotateObjects = true;
    }

    if(ImGui::RadioButton("lights", !m_rotateObjects))
    {
        m_rotateObjects = false;
    }
    ImGui::EndGroup();

    ImGui::SetNextItemWidth(150);
    if(ImGui::InputFloat("Ia (ambient)", &ia, 0.005f, 0.01f)) 
    {
        if (ia < 0.0f)
            ia = 0.0f;
        if (ia > 1.0f)
            ia = 1.0f;
    }

    ImGui::Text("Average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::Text("Press ESC to exit to ui");
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
            m_inputMouseLocked = true;
            if(m_rotateObjects) 
            {
                
                glfwSetCursorPosCallback(m_rendering.window, mouseCallbackRotateAll);
            }
            else 
            {
                glfwSetCursorPosCallback(m_rendering.window, mouseCallbackRotateLights);
            }
        }
    }
}

void imGuiUi::newFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}
