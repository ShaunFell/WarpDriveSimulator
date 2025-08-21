#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <string>

namespace WarpDrive {

class Shader {
public:
  unsigned int ID;

  Shader(const char *vertex_path, const char *fragment_path);
  Shader(const char *compute_path); // For compute shaders
  ~Shader();

  void use();
  void dispatch_compute(unsigned int num_groups_x, unsigned int num_groups_y,
                        unsigned int num_groups_z = 1);

  // Uniform setters
  void set_bool(const std::string &name, bool value) const;
  void set_int(const std::string &name, int value) const;
  void set_float(const std::string &name, float value) const;
  void set_vec2(const std::string &name, const glm::vec2 &value) const;
  void set_vec3(const std::string &name, const glm::vec3 &value) const;
  void set_vec4(const std::string &name, const glm::vec4 &value) const;
  void set_mat2(const std::string &name, const glm::mat2 &mat) const;
  void set_mat3(const std::string &name, const glm::mat3 &mat) const;
  void set_mat4(const std::string &name, const glm::mat4 &mat) const;

private:
  void check_compile_errors(unsigned int shader, std::string type);
  std::string load_shader_source(const char *path);
};

} // namespace WarpDrive
