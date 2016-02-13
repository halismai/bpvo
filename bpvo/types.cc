/*
   This file is part of bpvo.

   bpvo is free software: you can redistribute it and/or modify
   it under the terms of the Lesser GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   bpvo is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   Lesser GNU General Public License for more details.

   You should have received a copy of the Lesser GNU General Public License
   along with bpvo.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Contributor: halismai@cs.cmu.edu
 */

#include "bpvo/types.h"
#include "bpvo/utils.h"
#include "bpvo/config_file.h"
#include <iostream>
#include <string>

namespace bpvo {

AlgorithmParameters::AlgorithmParameters()
    : numPyramidLevels(-1)
    , sigmaPriorToCensusTransform(-1.0f)
    , sigmaBitPlanes(0.5)
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
    , goodPointThreshold(0.85)
    , minNumPixelsForNonMaximaSuppression(320*240)
    , minSaliency(0.1)
    , minDisparity(1.0)
    , maxDisparity(512.0f)
    , maxTestLevel(0)
    , withNormalization(true) {}


AlgorithmParameters::AlgorithmParameters(std::string filename)
{
  ConfigFile cf(filename);

  numPyramidLevels = cf.get<int>("numPyramidLevels", -1);
  sigmaPriorToCensusTransform = cf.get<float>("sigmaPriorToCensusTransform", 0.5f);
  sigmaBitPlanes = cf.get<float>("sigmaBitPlanes", 0.5f);
  maxIterations = cf.get<int>("maxIterations", 50);
  parameterTolerance = cf.get<float>("parameterTolerance", 1e-7);
  functionTolerance = cf.get<float>("functionTolerance", 1e-6);
  gradientTolerance = cf.get<float>("gradientTolerance", 1e-6);
  relaxTolerancesForCoarseLevels = cf.get<int>("relaxTolerancesForCoarseLevels", 1);
  lossFunction = LossFunctionTypeFromString(cf.get<std::string>("lossFunction", "Huber"));
  verbosity = VerbosityTypeFromString(cf.get<std::string>("Verbosity", "Iteration"));
  minTranslationMagToKeyFrame = cf.get<float>("minTranslationMagToKeyFrame", 0.1);
  minRotationMagToKeyFrame = cf.get<float>("minRotationMagToKeyFrame", 2.5);
  maxFractionOfGoodPointsToKeyFrame = cf.get<float>("maxFractionOfGoodPointsToKeyFrame", 0.6f);
  goodPointThreshold = cf.get<float>("goodPointThreshold", 0.75);
  minNumPixelsForNonMaximaSuppression = cf.get<int>("minNumPixelsForNonMaximaSuppression", 320*240);
  minSaliency = cf.get<float>("minSaliency", 0.1f);
  minDisparity = cf.get<float>("minDisparity", 1.0f);
  maxDisparity = cf.get<float>("maxDisparity", 512.0f);
  maxTestLevel = cf.get<int>("maxTestLevel", 0);
  withNormalization = cf.get<int>("withNormalization", true);
}


std::string ToString(LossFunctionType t)
{
  switch(t) {
    case kHuber: return "Huber";
    case kTukey: return "Tukey";
    case kL2: return "L2";
  }

  return "Unknown";
}

LossFunctionType LossFunctionTypeFromString(std::string s)
{
  if(icompare("Huber", s))
    return kHuber;
  else if(icompare("Tukey", s))
    return kTukey;
  else if(icompare("L2", s))
    return kL2;
  else
    THROW_ERROR("unknown LossFunctionType");
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

VerbosityType VerbosityTypeFromString(std::string s)
{
  if(icompare("Iteration", s))
    return kIteration;
  else if(icompare("Final", s))
    return kFinal;
  else if(icompare("Silent", s))
    return kSilent;
  else if(icompare("Debug", s))
    return kDebug;
  else
    THROW_ERROR("unknown VerbosityType");
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
    case KeyFramingReason::kLargeTranslation: return "LargeTranslation";
    case KeyFramingReason::kLargeRotation: return "LargeRotation";
    case KeyFramingReason::kSmallFracOfGoodPoints: return "SmallFracOfGoodPoints";
    case KeyFramingReason::kNoKeyFraming: return "NoKeyFraming";
    case KeyFramingReason::kFirstFrame: return "FirstFrame";
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
  os << "goodPointThreshold = " << p.goodPointThreshold << "\n";
  os << "minNumPixelsForNonMaximaSuppression = " << p.minNumPixelsForNonMaximaSuppression << "\n";
  os << "minSaliency = " << p.minSaliency << "\n";
  os << "minDisparity = " << p.minDisparity << "\n";
  os << "maxDisparity = " << p.maxDisparity << "\n";
  os << "withNormalization = " << p.withNormalization << "\n";
  os << "maxTestLevel = " << p.maxTestLevel;

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
  , covariance(PoseCovariance::Identity())
  , isKeyFrame(false)
  , keyFramingReason(kNoKeyFraming) {}

std::ostream& operator<<(std::ostream& os, const Result& r)
{
  os << r.pose << "\n";
  os << "isKeyFrame: " << std::boolalpha << r.isKeyFrame << std::noboolalpha << "\n";
  if(!r.optimizerStatistics.empty())
    os << r.optimizerStatistics.front();

  return os;
}

std::ostream& operator<<(std::ostream& os, const ImageSize& s)
{
  os << "[" << s.rows << "," << s.cols << "]";
  return os;
}

}

