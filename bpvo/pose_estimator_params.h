#ifndef BPVO_POSE_ESTIMATOR_PARAMS_H
#define BPVO_POSE_ESTIMATOR_PARAMS_H

#include <bpvo/types.h>
#include <iosfwd>

namespace bpvo {

struct PoseEstimatorParameters
{
  int maxIterations = 50;
  int maxFuncEvals = 6*200;
  float functionTolerance   = 1e-6;
  float parameterTolerance  = 1e-6;
  float gradientTolerance   = 1e-6;
  LossFunctionType lossFunction = LossFunctionType::kHuber;

  VerbosityType verbosity = VerbosityType::kSilent;

  /**
   */
  inline PoseEstimatorParameters() {}

  /**
   */
  explicit PoseEstimatorParameters(const AlgorithmParameters& p);

  /**
   */
  void relaxTolerance(int max_it = 42, float scale_by = 10.0f);

  /**
   */
  friend std::ostream& operator<<(std::ostream&, const PoseEstimatorParameters&);
}; // PoseEstimatorParameters


}; // bpvo

#endif // BPVO_POSE_ESTIMATOR_PARAMS_H
