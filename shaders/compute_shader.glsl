#version 430 core
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Geodesic ray structure
struct GeodesicRay {
    vec4 position;      // (t, x, y, z)
    vec4 four_velocity; // (dt/dτ, dx/dτ, dy/dτ, dz/dτ)
    float proper_time;
    bool active;
    vec3 color;
    float padding;
};

// Uniform buffer
layout(std140, binding = 0) uniform SpacetimeUBO {
    float grid_size;
    float time;
    float max_integration_time;
    int ray_count;
    
    float warp_speed;
    float bubble_thickness;
    vec3 warp_direction;
    
    mat4 view_matrix;
    mat4 projection_matrix;
    vec3 camera_position;
    float padding;
};

// Ray storage buffer
layout(std430, binding = 1) restrict buffer RayBuffer {
    GeodesicRay rays[];
};

// Spacetime metric evaluation (flat spacetime for Phase 1)
void evaluate_metric(vec3 pos, float t, out float lapse, out vec3 shift, out mat3 spatial_metric) {
    // Minkowski spacetime
    lapse = 1.0;
    shift = vec3(0.0);
    spatial_metric = mat3(1.0); // Identity matrix
}

// Geodesic equation integration (RK4)
void integrate_geodesic(inout GeodesicRay ray, float dt) {
    if (!ray.active) return;
    
    vec3 pos = ray.position.yzw;
    vec4 vel = ray.four_velocity;
    
    // For flat spacetime, geodesics are straight lines
    ray.position += vel * dt;
    ray.proper_time += dt;
    
    // Deactivate ray if it's gone too far
    if (ray.proper_time > max_integration_time || length(pos) > 100.0) {
        ray.active = false;
    }
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= ray_count) return;
    
    // Integration step size
    float dt = 0.01;
    
    // Integrate one step
    integrate_geodesic(rays[index], dt);
}
