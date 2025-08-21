#include "core/flat_spacetime.h"

namespace WarpDrive {

FlatSpacetime::FlatSpacetime() {
    time_scale = 1.0f;
    space_scale = 1.0f;
}

Spacetime3Plus1 FlatSpacetime::evaluate_metric(const glm::vec3& position, float time) const {
    Spacetime3Plus1 metric;
    
    // Minkowski spacetime in 3+1 form
    metric.lapse = 1.0f;
    metric.shift = glm::vec3(0.0f);
    
    // Flat spatial metric (Euclidean)
    metric.spatial_metric[0] = 1.0f; // γxx
    metric.spatial_metric[1] = 0.0f; // γxy
    metric.spatial_metric[2] = 0.0f; // γxz  
    metric.spatial_metric[3] = 1.0f; // γyy
    metric.spatial_metric[4] = 0.0f; // γyz
    metric.spatial_metric[5] = 1.0f; // γzz
    
    return metric;
}

glm::vec4 FlatSpacetime::compute_christoffel(const glm::vec3& position, float time, 
                                           int mu, int nu, int rho) const {
    // All Christoffel symbols are zero in flat spacetime
    return glm::vec4(0.0f);
}

} // namespace WarpDrive
