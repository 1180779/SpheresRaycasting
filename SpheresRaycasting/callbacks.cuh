
#ifndef U1180779_CALLBACKS_CUH
#define U1180779_CALLBACKS_CUH

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include "cudaWrappers.hpp"
#include "transformScene.hpp"
#include "rendering.hpp"
#include "lights.hpp"

extern transformData* spheresDataForCallback;
extern const bvhDevice* spheresBvhForCallback;
extern lights* lightsCallback;

static float lastX = 1280 / 2, lastY = 720 / 2;

/* mouse callbacks to rotate part of objects */

void mouseCallbackRotateAll(GLFWwindow* window, double xpos, double ypos);
void mouseCallbackRotateLights(GLFWwindow* window, double xpos, double ypos);

void disableCursor(const rendering& render);

#endif