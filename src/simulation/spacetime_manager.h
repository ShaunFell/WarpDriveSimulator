#pragma once
#include "core/spacetime_metric.h"
#include "core/types.h"
#include "rendering/camera.h"
#include "rendering/shader.h"
#include <GL/glew.h>
#include <memory>
#include <vector>

namespace WarpDrive {

class SpacetimeManager {
public:
  SpacetimeManager(int width, int height);
  ~SpacetimeManager();

  void initialize();
  void update(float delta_time);
  void render(const Camera &camera);
  void set_spacetime_metric(std::unique_ptr<SpacetimeMetric> metric);

  // Simulation parameters
  void set_ray_count(int count) { ray_count = count; }
  void set_integration_time(float time) { max_integration_time = time; }

  // Getters
  float get_simulation_time() const { return simulation_time; }
  int get_ray_count() const { return ray_count; }

private:
  void setup_buffers();
  void setup_shaders();
  void initialize_rays();
  void update_uniform_buffer(const Camera &camera);
  void dispatch_geodesic_compute();
  void render_rays();
  void render_background();

  // OpenGL resources
  unsigned int VAO, VBO, EBO;
  unsigned int ray_ssbo; // Ray storage buffer
  unsigned int ubo;      // Uniform buffer
  unsigned int background_texture;

  std::unique_ptr<Shader> compute_shader;
  std::unique_ptr<Shader> ray_shader;
  std::unique_ptr<Shader> background_shader;

  // Simulation state
  std::unique_ptr<SpacetimeMetric> current_metric;
  std::vector<GeodesicRay> geodesic_rays;
  SpacetimeUBO uniform_data;

  // Parameters
  int width, height;
  int ray_count = 1024;
  float max_integration_time = 100.0f;
  float simulation_time = 0.0f;

  // Background grid for spacetime visualization
  std::vector<glm::vec3> grid_vertices;
  std::vector<unsigned int> grid_indices;
};

} // namespace WarpDrive
