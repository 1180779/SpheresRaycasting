
#ifndef U1180779_IMGUIUI_H
#define U1180779_IMGUIUI_H

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include "rendering.hpp"
#include "imGui/imgui.h"
#include "imGui/imgui_impl_glfw.h"
#include "imGui/imgui_impl_opengl3.h"

class imGuiUi 
{
public:
    imGuiUi(rendering& rendering);
    ~imGuiUi();

    void styleRounded();
    void styleSquare();
    void styleLight();
    void styleDark();

    void settingsWindow();
    void newFrame();
    ImGuiIO& io;

private:
    rendering& m_rendering;
};

#endif