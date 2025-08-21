#pragma once
#include "spacetime_metric.h"

namespace WarpDrive {

class FlatSpacetime : public SpacetimeMetric {
public:
    FlatSpacetime();
    virtual ~FlatSpacetime() = default;
    
    Spacetime3Plus1 evaluate_metric(const glm::vec3& position, float time) const override;
    glm::vec4 compute_christoffel(const glm::vec3& position, float time, 
                                 int mu, int nu, int rho) const override;
};

} // namespace WarpDrive
