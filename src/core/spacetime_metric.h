#pragma once
#include "types.h"
#include <glm/glm.hpp>

namespace WarpDrive {

class SpacetimeMetric {
public:
    virtual ~SpacetimeMetric() = default;
    
    // Pure virtual methods for metric evaluation
    virtual Spacetime3Plus1 evaluate_metric(const glm::vec3& position, float time) const = 0;
    virtual glm::vec4 compute_christoffel(const glm::vec3& position, float time, 
                                         int mu, int nu, int rho) const = 0;
    
    // Utility methods
    virtual glm::mat4 get_four_metric(const glm::vec3& position, float time) const;
    virtual float compute_proper_distance(const glm::vec3& start, const glm::vec3& end, 
                                        float time) const;
    
    // Parameter getters/setters
    virtual void set_parameter(const std::string& name, float value) {}
    virtual float get_parameter(const std::string& name) const { return 0.0f; }
    
protected:
    float time_scale = 1.0f;
    float space_scale = 1.0f;
};

} // namespace WarpDrive
