#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace WarpDrive {

// 3+1 Decomposition structure
struct Spacetime3Plus1 {
    float lapse = 1.0f;                    // α(x,t)
    glm::vec3 shift = glm::vec3(0.0f);     // βⁱ(x,t)
    float spatial_metric[6] = {1,0,0,1,0,1}; // γᵢⱼ symmetric: xx,xy,xz,yy,yz,zz
    
    Spacetime3Plus1() {
        lapse = 1.0f;
        shift = glm::vec3(0.0f);
        // Identity spatial metric (flat space)
        spatial_metric[0] = 1.0f; // γxx
        spatial_metric[1] = 0.0f; // γxy
        spatial_metric[2] = 0.0f; // γxz
        spatial_metric[3] = 1.0f; // γyy
        spatial_metric[4] = 0.0f; // γyz
        spatial_metric[5] = 1.0f; // γzz
    }
};

// Geodesic ray for ray tracing
struct GeodesicRay {
    glm::vec4 position = glm::vec4(0.0f);      // (t, x, y, z)
    glm::vec4 four_velocity = glm::vec4(1,0,0,0); // (dt/dτ, dx/dτ, dy/dτ, dz/dτ)
    float proper_time = 0.0f;
    bool active = true;
    glm::vec3 color = glm::vec3(1.0f);         // For visualization
};

// Uniform buffer for GPU compute
struct SpacetimeUBO {
    float grid_size = 128.0f;
    float time = 0.0f;
    float max_integration_time = 100.0f;
    int ray_count = 1024;
    
    // Spacetime parameters (will be expanded for warp drive)
    float warp_speed = 0.0f;
    float bubble_thickness = 1.0f;
    glm::vec3 warp_direction = glm::vec3(0,0,1);
    
    // Camera parameters
    glm::mat4 view_matrix = glm::mat4(1.0f);
    glm::mat4 projection_matrix = glm::mat4(1.0f);
    glm::vec3 camera_position = glm::vec3(0.0f);
    float padding = 0.0f;
};

} // namespace WarpDrive
