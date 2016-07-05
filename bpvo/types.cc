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
#include "bpvo/point_cloud.h"
#include <iostream>
#include <string>

namespace bpvo {

AlgorithmParameters::AlgorithmParameters()
    : numPyramidLevels(-1)
    , minImageDimensionForPyramid(40)
    , sigmaPriorToCensusTransform(-1.0f)
    , sigmaBitPlanes(0.5)
    , dfSigma1(0.75)
    , dfSigma2(1.75)
    , latchNumBytes(1)
    , latchRotationInvariance(false)
    , latchHalfSsdSize(1)
    , centralDifferenceRadius(3)
    , centralDifferenceSigmaBefore(0.75)
    , centralDifferenceSigmaAfter(1.75)
    , laplacianKernelSize(1)
    , maxIterations(50)
    , parameterTolerance(1e-7)
    , functionTolerance(1e-6)
    , gradientTolerance(1e-8)
    , relaxTolerancesForCoarseLevels(true)
    , gradientEstimation(GradientEstimationType::kCentralDifference_3)
    , interp(InterpolationType::kLinear)
    , lossFunction(LossFunctionType::kTukey)
    , descriptor(DescriptorType::kIntensity)
    , verbosity(VerbosityType::kIteration)
    , minTranslationMagToKeyFrame(0.15)
    , minRotationMagToKeyFrame(5.0)
    , maxFractionOfGoodPointsToKeyFrame(0.6)
    , goodPointThreshold(0.85)
    , minNumPixelsForNonMaximaSuppression(320*240)
    , nonMaxSuppRadius(1)
    , minNumPixelsToWork(256)
    , minSaliency(0.1)
    , minValidDisparity(0.001)
    , maxValidDisparity(512.0f)
    , maxTestLevel(0)
    , withNormalization(true) {}

AlgorithmParameters::AlgorithmParameters(std::string filename)
{
  ConfigFile cf(filename);

  numPyramidLevels = cf.get<int>("numPyramidLevels", -1);
  minImageDimensionForPyramid = cf.get<int>("minImageDimensionForPyramid", 40);
  sigmaPriorToCensusTransform = cf.get<float>("sigmaPriorToCensusTransform", 0.5f);
  sigmaBitPlanes = cf.get<float>("sigmaBitPlanes", 0.5f);
  dfSigma1 = cf.get<float>("dfSigma1", 0.75);
  dfSigma2 = cf.get<float>("dfSigma2", 1.75);
  latchNumBytes = cf.get<int>("latchNumBytes", 1);
  latchRotationInvariance = cf.get<int>("latchRotationInvariance", 0);
  latchHalfSsdSize = cf.get<int>("latchHalfSsdSize", 1);
  centralDifferenceRadius = cf.get<int>("centralDifferenceRadius", 3);
  centralDifferenceSigmaBefore = cf.get<float>("centralDifferenceSigmaBefore", 0.75);
  centralDifferenceSigmaAfter = cf.get<float>("CenteralDifferenceSigmaAfter", 1.75);
  laplacianKernelSize = cf.get<int>("laplacianKernelSize", 1);
  maxIterations = cf.get<int>("maxIterations", 50);
  parameterTolerance = cf.get<float>("parameterTolerance", 1e-7);
  functionTolerance = cf.get<float>("functionTolerance", 1e-6);
  gradientTolerance = cf.get<float>("gradientTolerance", 1e-6);
  relaxTolerancesForCoarseLevels = cf.get<int>("relaxTolerancesForCoarseLevels", 1);
  gradientEstimation = GradientEstimationTypeFromString(cf.get<std::string>("GradientEstimation", "CD5"));
  interp = InterpolationTypeFromString(cf.get<std::string>("Interpolation", "Linear"));
  lossFunction = LossFunctionTypeFromString(cf.get<std::string>("lossFunction", "Huber"));
  descriptor = DescriptorTypeFromString(cf.get<std::string>("descriptor", "Intensity"));
  verbosity = VerbosityTypeFromString(cf.get<std::string>("Verbosity", "Iteration"));
  minTranslationMagToKeyFrame = cf.get<float>("minTranslationMagToKeyFrame", 0.1);
  minRotationMagToKeyFrame = cf.get<float>("minRotationMagToKeyFrame", 2.5);
  maxFractionOfGoodPointsToKeyFrame = cf.get<float>("maxFractionOfGoodPointsToKeyFrame", 0.6f);
  goodPointThreshold = cf.get<float>("goodPointThreshold", 0.75);
  minNumPixelsForNonMaximaSuppression = cf.get<int>("minNumPixelsForNonMaximaSuppression", 320*240);
  nonMaxSuppRadius = cf.get<int>("nonMaxSuppRadius", 1);
  minNumPixelsToWork = cf.get<int>("minNumPixelsToWork", 256);
  minSaliency = cf.get<float>("minSaliency", 0.1f);
  minValidDisparity = cf.get<float>("minValidDisparity", 1.0f);
  maxValidDisparity = cf.get<float>("maxValidDisparity", 512.0f);
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

InterpolationType InterpolationTypeFromString(std::string s)
{
  if(icompare("Linear", s))
    return kLinear;
  else if(icompare("Cosine", s))
    return kCosine;
  else if(icompare("CubicHermite", s))
    return kCubicHermite;
  else if(icompare("Cubic", s))
    return kCubic;
  else
    THROW_ERROR("unknown InterpolationType");
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

DescriptorType DescriptorTypeFromString(std::string s)
{
  if(icompare("Intensity", s))
    return kIntensity;
  else if(icompare("BitPlanes", s))
    return kBitPlanes;
  else if(icompare("Gradient", s) || icompare("IntensityAndGradient", s))
    return kIntensityAndGradient;
  else if(icompare("DescriptorFields", s))
    return kDescriptorFieldsFirstOrder;
  else if(icompare("Latch", s))
    return kLatch;
  else if(icompare("CentralDifference", s))
    return kCentralDifference;
  else if(icompare("Laplacian", s))
    return kLaplacian;
  else
    THROW_ERROR(Format("unknown DescriptorType %s", s.c_str()).c_str());
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

GradientEstimationType GradientEstimationTypeFromString(std::string s)
{
  if(icompare("CD3", s))
    return kCentralDifference_3;
  else if(icompare("CD5", s))
    return kCentralDifference_5;
  else
    THROW_ERROR("unknown GradientEstimationType");
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

std::string ToString(DescriptorType t)
{
  switch(t) {
    case DescriptorType::kIntensity: return "Intensity"; break;
    case DescriptorType::kIntensityAndGradient: return "IntensityAndGradient"; break;
    case DescriptorType::kDescriptorFieldsFirstOrder: return "DescriptorFields"; break;
    case DescriptorType::kDescriptorFieldsSecondOrder: return "DescriptorFields2ndOrder"; break;
    case DescriptorType::kBitPlanes: return "BitPlanes"; break;
    case DescriptorType::kLatch: return "Latch"; break;
    case DescriptorType::kCentralDifference: return "CenteralDifference"; break;
    case DescriptorType::kLaplacian: return "Laplacian"; break;
  }

  return "Unknown";
}

std::string ToString(GradientEstimationType t)
{
  switch(t) {
    case GradientEstimationType::kCentralDifference_3: return "CentralDifference_3"; break;
    case GradientEstimationType::kCentralDifference_5: return "CentralDifference_5"; break;
  }

  return "Unknown";
}

std::string ToString(InterpolationType t)
{
  switch(t) {
    case InterpolationType::kLinear: return "Linear"; break;
    case InterpolationType::kCosine: return "Cosine"; break;
    case InterpolationType::kCubic: return "Cubic"; break;
    case InterpolationType::kCubicHermite: return "CubicHermite"; break;
  }

  return "Unknown";
}


std::ostream& operator<<(std::ostream& os, const AlgorithmParameters& p)
{
  os << "numPyramidLevels = " << p.numPyramidLevels << "\n";
  os << "minImageDimensionForPyramid = " << p.minImageDimensionForPyramid << "\n";
  os << "sigmaPriorToCensusTransform = " << p.sigmaPriorToCensusTransform << "\n";
  os << "sigmaBitPlanes = " << p.sigmaBitPlanes << "\n";
  os << "dfSigma1 = " << p.dfSigma1 << "\n";
  os << "dfSigma2 = " << p.dfSigma2 << "\n";
  os << "latchNumBytes = " << p.latchNumBytes << "\n";
  os << "latchRotationInvariance = " << p.latchRotationInvariance << "\n";
  os << "latchHalfSsdSize = " << p.latchHalfSsdSize << "\n";
  os << "centralDifferenceRadius = " << p.centralDifferenceRadius << "\n";
  os << "centralDifferenceSigmaBefore = " << p.centralDifferenceSigmaBefore << "\n";
  os << "centralDifferenceSigmaAfter = " << p.centralDifferenceSigmaAfter << "\n";
  os << "laplacianKernelSize = " << p.laplacianKernelSize << "\n";
  os << "maxIterations = " << p.maxIterations << "\n";
  os << "parameterTolerance = " << p.parameterTolerance << "\n";
  os << "functionTolerance = " << p.functionTolerance << "\n";
  os << "gradientTolerance = " << p.gradientTolerance << "\n";
  os << "relaxTolerancesForCoarseLevel = " << p.relaxTolerancesForCoarseLevels << "\n";
  os << "gradienEstimation: " << ToString(p.gradientEstimation) << "\n";
  os << "InterpolationType: " << ToString(p.interp) << "\n";
  os << "lossFunction = " << ToString(p.lossFunction) << "\n";
  os << "verbosity = " << ToString(p.verbosity) << "\n";
  os << "minTranslationMagToKeyFrame = " << p.minTranslationMagToKeyFrame << "\n";
  os << "minRotationMagToKeyFrame = " << p.minRotationMagToKeyFrame << "\n";
  os << "maxFractionOfGoodPointsToKeyFrame = " << p.maxFractionOfGoodPointsToKeyFrame << "\n";
  os << "goodPointThreshold = " << p.goodPointThreshold << "\n";
  os << "minNumPixelsForNonMaximaSuppression = " << p.minNumPixelsForNonMaximaSuppression << "\n";
  os << "minNumPixelsToWork = " << p.minNumPixelsToWork << "\n";
  os << "minSaliency = " << p.minSaliency << "\n";
  os << "minValidDisparity = " << p.minValidDisparity << "\n";
  os << "maxValidDisparity = " << p.maxValidDisparity << "\n";
  os << "withNormalization = " << p.withNormalization << "\n";
  os << "maxTestLevel = " << p.maxTestLevel;

  return os;
}

OptimizerStatistics::OptimizerStatistics()
  : numIterations(0)
  , finalError(-1.0f)
  , firstOrderOptimality(-1.0f)
  , status(PoseEstimationStatus::kSolverError) {}

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

Result::Result(Result&& other) noexcept
  : pose(other.pose)
  , covariance(other.covariance)
  , optimizerStatistics(std::move(other.optimizerStatistics))
  , isKeyFrame(other.isKeyFrame)
  , keyFramingReason(other.keyFramingReason)
  , pointCloud(std::move(other.pointCloud)) {}

Result::~Result() {}

Result& Result::operator=(Result&& r) noexcept
{
  pose = r.pose;
  covariance = r.covariance;
  optimizerStatistics = std::move(r.optimizerStatistics);
  isKeyFrame = r.isKeyFrame;
  keyFramingReason = r.keyFramingReason;
  pointCloud = std::move(r.pointCloud);
  return *this;
}

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

} // bpvo

