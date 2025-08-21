#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <memory>

#include "core/flat_spacetime.h"
#include "rendering/camera.h"
#include "simulation/spacetime_manager.h"

using namespace WarpDrive;

// Settings
const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;

// Global variables
Camera camera(45.0f, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 1000.0f);
bool first_mouse = true;
float last_x = SCR_WIDTH / 2.0f;
float last_y = SCR_HEIGHT / 2.0f;
float delta_time = 0.0f;
float last_frame = 0.0f;

// Callback functions
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
void process_input(GLFWwindow *window);

int main() {
  // Initialize GLFW
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Create window
  GLFWwindow *window = glfwCreateWindow(
      SCR_WIDTH, SCR_HEIGHT, "Warp Drive Spacetime Simulator", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);

  // Capture mouse
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  if (glewInit() != GLEW_OK) {
    std::cout << "Failed to initialize GLEW" << std::endl;
    return -1;
  }
  // Configure OpenGL
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Create spacetime manager
  SpacetimeManager spacetime_manager(SCR_WIDTH, SCR_HEIGHT);

  // Set initial spacetime (flat/Minkowski)
  auto flat_spacetime = std::make_unique<FlatSpacetime>();
  spacetime_manager.set_spacetime_metric(std::move(flat_spacetime));

  // Initialize simulation
  spacetime_manager.initialize();

  std::cout << "Warp Drive Spacetime Simulator - Phase 1" << std::endl;
  std::cout << "Controls:" << std::endl;
  std::cout << "  WASD - Move camera" << std::endl;
  std::cout << "  QE - Move up/down" << std::endl;
  std::cout << "  Mouse - Look around" << std::endl;
  std::cout << "  Scroll - Zoom" << std::endl;
  std::cout << "  ESC - Exit" << std::endl;

  // Render loop
  while (!glfwWindowShouldClose(window)) {
    // Per-frame time logic
    float current_frame = glfwGetTime();
    delta_time = current_frame - last_frame;
    last_frame = current_frame;

    // Input
    process_input(window);

    // Update simulation
    spacetime_manager.update(delta_time);

    // Render
    glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    spacetime_manager.render(camera);

    // Swap buffers and poll events
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // Cleanup
  glfwTerminate();
  return 0;
}

void process_input(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  // Camera movement
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.process_keyboard(GLFW_KEY_W, delta_time);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.process_keyboard(GLFW_KEY_S, delta_time);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.process_keyboard(GLFW_KEY_A, delta_time);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.process_keyboard(GLFW_KEY_D, delta_time);
  if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    camera.process_keyboard(GLFW_KEY_E, delta_time);
  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    camera.process_keyboard(GLFW_KEY_Q, delta_time);
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
  camera.aspect_ratio = (float)width / (float)height;
}

void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
  if (first_mouse) {
    last_x = xpos;
    last_y = ypos;
    first_mouse = false;
  }

  float xoffset = xpos - last_x;
  float yoffset = last_y - ypos;

  last_x = xpos;
  last_y = ypos;

  camera.process_mouse(xoffset, yoffset);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  camera.process_mouse_scroll(yoffset);
}
