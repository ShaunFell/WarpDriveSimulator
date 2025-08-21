#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace WarpDrive {

class Camera {
public:
    Camera(float fov = 45.0f, float aspect = 16.0f/9.0f, float near = 0.1f, float far = 1000.0f);
    
    // Movement
    void process_keyboard(int key, float delta_time);
    void process_mouse(float xoffset, float yoffset, bool constrain_pitch = true);
    void process_mouse_scroll(float yoffset);
    
    // Getters
    glm::mat4 get_view_matrix() const;
    glm::mat4 get_projection_matrix() const;
    glm::vec3 get_position() const { return position; }
    glm::vec3 get_front() const { return front; }
    glm::vec3 get_up() const { return up; }
    glm::vec3 get_right() const { return right; }
    
    // Camera parameters
    float fov;
    float aspect_ratio;
    float near_plane;
    float far_plane;
    
private:
    void update_camera_vectors();
    
    // Camera attributes
    glm::vec3 position = glm::vec3(0.0f, 0.0f, 5.0f);
    glm::vec3 front = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right = glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 world_up = glm::vec3(0.0f, 1.0f, 0.0f);
    
    // Euler angles
    float yaw = -90.0f;
    float pitch = 0.0f;
    
    // Camera options
    float movement_speed = 10.0f;
    float mouse_sensitivity = 0.1f;
    float zoom = 45.0f;
};

} // namespace WarpDrive
