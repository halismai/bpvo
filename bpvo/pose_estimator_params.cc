#include "bpvo/pose_estimator_params.h"
#include <iostream>

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

std::ostream& operator<<(std::ostream& os, const PoseEstimatorParameters& p)
{
  os << "maxIterations: " << p.maxIterations << "\n";
  os << "functionTolerance: " << p.functionTolerance << "\n";
  os << "parameterTolerance: " << p.parameterTolerance << "\n";
  os << "gradientTolerance: " << p.gradientTolerance << "\n";
  os << "lossFunction: " << ToString(p.lossFunction) << "\n";
  os << "verbosity: " << ToString(p.verbosity);
  return os;
}

}; // bpvo

