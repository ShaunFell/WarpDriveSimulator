#include "rendering/camera.h"
#include <GLFW/glfw3.h>
#include <algorithm>

namespace WarpDrive {

Camera::Camera(float fov, float aspect, float near, float far) 
    : fov(fov), aspect_ratio(aspect), near_plane(near), far_plane(far) {
    update_camera_vectors();
}

void Camera::process_keyboard(int key, float delta_time) {
    float velocity = movement_speed * delta_time;
    
    if (key == GLFW_KEY_W)
        position += front * velocity;
    if (key == GLFW_KEY_S)
        position -= front * velocity;
    if (key == GLFW_KEY_A)
        position -= right * velocity;
    if (key == GLFW_KEY_D)
        position += right * velocity;
    if (key == GLFW_KEY_E)
        position += up * velocity;
    if (key == GLFW_KEY_Q)
        position -= up * velocity;
}

void Camera::process_mouse(float xoffset, float yoffset, bool constrain_pitch) {
    xoffset *= mouse_sensitivity;
    yoffset *= mouse_sensitivity;
    
    yaw += xoffset;
    pitch += yoffset;
    
    if (constrain_pitch) {
        pitch = std::clamp(pitch, -89.0f, 89.0f);
    }
    
    update_camera_vectors();
}

void Camera::process_mouse_scroll(float yoffset) {
    zoom -= yoffset;
    zoom = std::clamp(zoom, 1.0f, 45.0f);
}

glm::mat4 Camera::get_view_matrix() const {
    return glm::lookAt(position, position + front, up);
}

glm::mat4 Camera::get_projection_matrix() const {
    return glm::perspective(glm::radians(zoom), aspect_ratio, near_plane, far_plane);
}

void Camera::update_camera_vectors() {
    glm::vec3 new_front;
    new_front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    new_front.y = sin(glm::radians(pitch));
    new_front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(new_front);
    
    right = glm::normalize(glm::cross(front, world_up));
    up = glm::normalize(glm::cross(right, front));
}

} // namespace WarpDrive
