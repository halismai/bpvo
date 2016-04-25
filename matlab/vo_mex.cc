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

#include <mexmat.h>

#if !defined(MEXMAT_WITH_OPENCV) || !defined(MEXMAT_WITH_EIGEN)
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
  mexError("compile with opencv and eigen\n");
}
#else

#include <bpvo/vo.h>
#include <bpvo/point_cloud.h>
#include <bpvo/utils.h>

#include <memory>
#include <string>
#include <vector>

template <typename T> static inline
T GetOption(const mex::Struct& s, std::string name, T default_val)
{
  return s.hasField(name) ? mex::getNumber<T>(s[name]) : default_val;
}

template <> inline
std::string GetOption(const mex::Struct& s, std::string name, std::string default_val)
{
  return s.hasField(name) ? mex::getString(s[name]) : default_val;
}

static inline bpvo::LossFunctionType StringToLossFunctionType(std::string s)
{
  if(bpvo::icompare("Huber", s))
    return bpvo::LossFunctionType::kHuber;
  else if(bpvo::icompare("tukey", s))
    return bpvo::LossFunctionType::kTukey;
  else if(bpvo::icompare("L2", s))
    return bpvo::LossFunctionType::kL2;
  else
    mexError("bad loss function %s\n", s.c_str());

  return bpvo::LossFunctionType::kHuber; // -Wreturn-value
}

static inline bpvo::VerbosityType StringToVerbosityType(std::string s)
{
  if(bpvo::icompare("Silent", s))
    return bpvo::VerbosityType::kSilent;
  else if(bpvo::icompare("Iteration", s))
    return bpvo::VerbosityType::kIteration;
  else if(bpvo::icompare("Final", s))
    return bpvo::VerbosityType::kFinal;
  else if(bpvo::icompare("kDebug", s))
    return bpvo::VerbosityType::kDebug;
  else
    mexError("unknown VerbosityType %s\n", s.c_str());

  return bpvo::VerbosityType::kSilent;
}

static inline bpvo::DescriptorType StringToDescriptorType(std::string s)
{
  if(bpvo::icompare("Intensity", s))
    return bpvo::DescriptorType::kIntensity;
  else if(bpvo::icompare("bitplanes", s))
    return bpvo::DescriptorType::kBitPlanes;
  else
    mexError("unkonwn DescriptorType %s\n", s.c_str());

  return bpvo::DescriptorType::kIntensity;
}

static inline
mex::Struct PointCloudToMatalb(const bpvo::UniquePointer<bpvo::PointCloud>& point_cloud)
{
  // X 3D points
  // C RGB color
  // w point weight
  // T the pose
  mex::Struct ret(std::vector<std::string>{"X", "C", "w", "T"});
  if(point_cloud) {
    const auto n = point_cloud->size();
    mex::Mat<float> X(3, n);
    mex::Mat<uint8_t> C(3, n);
    mex::Mat<float> w(1, n);
    for(size_t i = 0; i < point_cloud->size(); ++i) {
      const auto& p = point_cloud->operator[](i);
      memcpy(X.col(i), p.xyzw().data(), 3*sizeof(float));
      memcpy(C.col(i), p.rgba().data(), 3*sizeof(uint8_t));
      w[i] = p.weight();
    }

    ret.set("X", X);
    ret.set("C", C);
    ret.set("w", w);
    ret.set("T", mex::Mat<float>(point_cloud->pose()));
  }

  return ret;
}

static inline bpvo::AlgorithmParameters ToAlgorithmParameters(const mex::Struct& params)
{
  bpvo::AlgorithmParameters ret;

  ret.numPyramidLevels            = GetOption<int>(params, "numPyramidLevels", -1);
  ret.minImageDimensionForPyramid = GetOption<int>(params, "minImageDimensionForPyramid", 40);

  ret.sigmaPriorToCensusTransform = GetOption<float>(params, "sigmaPriorToCensusTransform", 0.5f);
  ret.sigmaBitPlanes              = GetOption<float>(params, "sigmaBitPlanes", 1.0f);

  ret.maxIterations       = GetOption<int>(params, "maxIterations", 100);
  ret.parameterTolerance  = GetOption<float>(params, "parameterTolerance", 1e-6);
  ret.functionTolerance   = GetOption<float>(params, "functionTolerance", 1e-5);
  ret.gradientTolerance   = GetOption<float>(params, "gradientTolerance", 1e-7);

  ret.relaxTolerancesForCoarseLevels = GetOption<int>(params,"relaxTolerancesForCoarseLevels", 1);

  ret.lossFunction = StringToLossFunctionType(GetOption<std::string>(params,"LossFunction", "Huber"));

  ret.verbosity = StringToVerbosityType(GetOption<std::string>(params,"verbosity", "Silent"));

  ret.minTranslationMagToKeyFrame = GetOption<float>(params, "minTranslationMagToKeyFrame", 0.1f);
  ret.minRotationMagToKeyFrame    = GetOption<float>(params, "minRotationMagToKeyFrame", 2.5f);
  ret.maxFractionOfGoodPointsToKeyFrame = GetOption<float>(params, "maxFractionOfGoodPointsToKeyFrame", 0.5f);
  ret.goodPointThreshold                = GetOption<float>(params, "goodPointThreshold", 0.9f);

  ret.nonMaxSuppRadius = GetOption<int>(params, "nonMaxSuppRadius", 1);
  ret.minNumPixelsForNonMaximaSuppression = GetOption<int>(params, "minNumPixelsForNonMaximaSuppression", 320*240);

  ret.maxTestLevel = GetOption<int>(params, "maxTestLevel", 0);

  ret.descriptor = StringToDescriptorType(
      GetOption<std::string>(params, "descriptor", "intensity"));

  ret.minSaliency = GetOption<float>(params, "minSaliency", 0.01);

  ret.minValidDisparity = GetOption<float>(params, "minValidDisparity", 0.001f);
  ret.maxValidDisparity = GetOption<float>(params, "maxValidDisparity", 1024.0f);

  ret.withNormalization = GetOption<int>(params, "withNormalization", 1);

  return ret;
}

static inline mex::Struct ResultToMex(const bpvo::Result result)
{
  const std::vector<std::string> fields{
    "pose", "covariance", "isKeyFrame", "optimizerStatistics", "keyFramingReason", "pointCloud"};
  mex::Struct ret(fields);

  ret.set("pose", mex::Mat<float>(result.pose));
  ret.set("covariance", mex::Mat<float>(result.covariance));
  ret.set("isKeyFrame", result.isKeyFrame);
  ret.set("keyFramingReason", bpvo::ToString(result.keyFramingReason));

  mex::Struct stats(std::vector<std::string>{
                    "numIterations", "finalError", "firstOrderOptimality", "status"},
                    1, result.optimizerStatistics.size());
  for(size_t i = 0; i < result.optimizerStatistics.size(); ++i) {
    const auto& s = result.optimizerStatistics[i];
    stats.set("numIterations", s.numIterations, i);
    stats.set("finalError", s.finalError, i);
    stats.set("firstOrderOptimality", s.firstOrderOptimality,i);
    stats.set("status", bpvo::ToString(s.status));
  }

  ret.set("optimizerStatistics", stats.release());
  ret.set("pointCloud", PointCloudToMatalb(result.pointCloud).release());

  return ret;
}

class VisualOdometryWrapper
{
  typedef bpvo::VisualOdometry Vo_t;

 public:
  inline VisualOdometryWrapper(const mex::Mat<double>& K, double b, int rows, int cols,
                               const mex::Struct& params)
      : _vo(new Vo_t(K.toEigenFixed<3,3>().cast<float>(), b, bpvo::ImageSize(rows, cols),
                     ToAlgorithmParameters(params))) {}

  inline mex::Struct addFrame(const mex::Mat<uint8_t>& I, const mex::Mat<float>& D)
  {
    const cv::Mat I_cv = mex::mex2cv<uint8_t>(I), D_cv = mex::mex2cv<float>(D);
    return ResultToMex(_vo->addFrame(I_cv.ptr<uint8_t>(), D_cv.ptr<float>()));
  }

 private:
  std::unique_ptr<Vo_t> _vo;
};

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
  if(nrhs < 1)
    mexError("need command\n");

  const std::string command(mex::getString(prhs[0]));
  if(bpvo::icompare("new", command))
  {
    static const char* USAGE = "hdle = fn('new', K, b, image_size, params)";
    mex::nargchk(5, 5, nrhs, USAGE);
    const mex::Mat<double> K(prhs[1]);
    const double b = mex::getNumber<double>(prhs[2]);
    const mex::Mat<double> image_size(prhs[3]);
    const mex::Struct params(prhs[4]);
    const int rows = static_cast<int>( image_size[0] );
    const int cols = static_cast<int>( image_size[1] );
    plhs[0] = mex::PtrToMex<VisualOdometryWrapper>(new VisualOdometryWrapper(K, b, rows, cols, params));
  }
  else if(bpvo::icompare("delete", command))
  {
    static const char* USAGE = "fn('delete', hdle)";
    mex::nargchk(2, 2, nrhs, USAGE);
    mex::nargchk(0, 0, nlhs, USAGE);
    mex::DeleteClass<VisualOdometryWrapper>(prhs[1]);
  }
  else if(bpvo::icompare("add_frame", command))
  {
    static const char* USAGE = "result = fn('add_frame', hdle, I, D)";
    mex::nargchk(4, 4, nrhs, USAGE);
    const mex::Mat<uint8_t> I(prhs[2]);
    const mex::Mat<float> D(prhs[3]);
    plhs[0] = mex::MexToPtr<VisualOdometryWrapper>(prhs[1])->addFrame(I, D).release();
  }
  else
  {
    mexError("unkown command %s\n", command.c_str());
  }
}

#endif //


