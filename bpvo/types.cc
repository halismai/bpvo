#include "bpvo/types.h"
#include <iostream>
#include <string>

namespace bpvo {

AlgorithmParameters::AlgorithmParameters()
    : numPyramidLevels(-1)
    , sigmaPriorToCensusTransform(-1.0f)
    , sigmaBitPlanes(1.618)
    , maxIterations(50)
    , parameterTolerance(1e-7)
    , functionTolerance(1e-6)
    , gradientTolerance(1e-8)
    , relaxTolerancesForCoarseLevels(true)
    , lossFunction(LossFunctionType::kTukey)
    , verbosity(VerbosityType::kIteration)
    , minTranslationMagToKeyFrame(0.15)
    , minRotationMagToKeyFrame(5.0)
    , maxFractionOfGoodPointsToKeyFrame(0.6)
    , goodPointThreshold(0.85) {}

std::string ToString(LossFunctionType t)
{
  switch(t) {
    case kHuber: return "Huber";
    case kTukey: return "Tukey";
    case kL2: return "L2";
  }

  return "Unknown";
}

std::string ToString(VerbosityType v)
{
  switch(v) {
    case kIteration: return "Iteration";
    case kFinal: return "Final";
    case kSilent: return "Silent";
    case kDebug: return "Debug";
  }

  return "Unknown";
}

std::string ToString(PoseEstimationStatus s)
{
  switch(s) {
    case kParameterTolReached: return "ParameterTolReached";
    case kFunctionTolReached: return "FunctionTolReached";
    case kGradientTolReached: return "GradientTolReached";
    case kMaxIterations: return "MaxIterations";
    case kSolverError: return "SolverError";
  }

  return "Unknown";
}

std::string ToString(KeyFramingReason r)
{
  switch(r) {
    case kLargeTranslation: return "LargeTranslation";
    case kLargeRotation: return "LargeRotation";
    case kSmallFracOfGoodPoints: return "SmallFracOfGoodPoints";
    case kNoKeyFraming: return "NoKeyFraming";
  }

  return "Unknown";
}

std::ostream& operator<<(std::ostream& os, const AlgorithmParameters& p)
{
  os << "numPyramidLevels = " << p.numPyramidLevels << "\n";
  os << "sigmaPriorToCensusTransform = " << p.sigmaPriorToCensusTransform << "\n";
  os << "sigmaBitPlanes = " << p.sigmaBitPlanes << "\n";
  os << "maxIterations = " << p.maxIterations << "\n";
  os << "parameterTolerance = " << p.parameterTolerance << "\n";
  os << "functionTolerance = " << p.functionTolerance << "\n";
  os << "gradientTolerance = " << p.gradientTolerance << "\n";
  os << "relaxTolerancesForCoarseLevel = " << p.relaxTolerancesForCoarseLevels << "\n";
  os << "lossFunction = " << ToString(p.lossFunction) << "\n";
  os << "verbosity = " << ToString(p.verbosity) << "\n";
  os << "minTranslationMagToKeyFrame = " << p.minTranslationMagToKeyFrame << "\n";
  os << "minRotationMagToKeyFrame = " << p.minRotationMagToKeyFrame << "\n";
  os << "maxFractionOfGoodPointsToKeyFrame = " << p.maxFractionOfGoodPointsToKeyFrame << "\n";
  os << "goodPointThreshold = " << p.goodPointThreshold;

  return os;
}

std::ostream& operator<<(std::ostream& os, const OptimizerStatistics& s)
{
  os << "numIterations: " << s.numIterations << "\n"
     << "finalError: " << s.finalError << "\n"
     << "firstOrderOptimality: " << s.firstOrderOptimality << "\n"
     << "status: " << ToString(s.status);

  return os;
}

Result::Result()
  : pose(Pose::Identity())
  , covriance(PoseCovariance::Identity())
  , isKeyFrame(false) {}

}
