#include "bpvo/pose_estimator.h"

namespace bpvo {

PoseEstimatorParameters::PoseEstimatorParameters(const AlgorithmParameters& p)
    : maxIterations(p.maxIterations)
        , functionTolerance(p.functionTolerance)
        , parameterTolerance(p.parameterTolerance)
        , gradientTolerance(p.gradientTolerance)
        , lossFunction(p.lossFunction)
        , verbosity(p.verbosity) {}


void PoseEstimatorParameters::relaxTolerance(int max_it, float scale_by)
{
  maxIterations = std::min(maxIterations, max_it);
  functionTolerance *= scale_by;
  parameterTolerance *= scale_by;
  gradientTolerance *= scale_by;

}

}; // bpvo
