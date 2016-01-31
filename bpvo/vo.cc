#include "bpvo/vo.h"
#include "bpvo/debug.h"
#include "bpvo/template_data.h"
#include "bpvo/linear_system_builder.h"
#include "bpvo/math_utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <memory>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Cholesky>

namespace bpvo {

static int getNumberOfPyramidLevels(int min_image_dim, int min_allowed_res)
{
  return 1 + std::round(std::log2(min_image_dim / (double) min_allowed_res));
}

template <typename T> static inline
cv::Mat ToOpenCV(const T* data, int rows, int cols)
{
  return cv::Mat(rows, cols, data, cv::DataType<T>::type);
}

struct KeyFrameCandidate
{
  std::vector<cv::Mat> image_pyramid;
  cv::Mat disparity;
}; // KeyFrameCandidate

struct PoseEstimator
{
 public:
  PoseEstimator(AlgorithmParameters params)
      : _params(params), _sys_builder(_params.lossFunction) {}

  OptimizerStatistics run(TemplateData* data, const cv::Mat& image, Matrix44& T)
  {
    data->setInputImage(image);

    static const char* fmt_str = "%3d        %13.6g  %12.3g    %12.6g   %12.6g\n";
    static constexpr float sqrtEps = std::sqrt(std::numeric_limits<float>::epsilon());

    if(_params.verbosity == VerbosityType::kIteration) {
      printf("\n                            1st-order     norm of           delta\n");
      printf(" Iteration      Residual    Optimality     step             error\n");
    }

    OptimizerStatistics ret;
    ret.numIterations = 0;
    ret.firstOrderOptimality = 0;

    typename LinearSystemBuilder::Hessian A;
    typename LinearSystemBuilder::Gradient b;
    typename LinearSystemBuilder::Gradient dp;
    float norm_e_prev = 0.0f; //std::numeric_limits<float>::max();

    while(ret.numIterations++ < _params.maxIterations) {
      data->computeResiduals(T, _residuals, _valid);
      ret.finalError = _sys_builder.run(data->jacobians(), _residuals, _valid, A, b);
      ret.firstOrderOptimality = b.lpNorm<Eigen::Infinity>();
      dp = A.ldlt().solve(b); // TODO check the solution

      auto norm_dp = dp.squaredNorm(); // could use squaredNorm
      auto norm_e  = ret.finalError;
      auto norm_g  = ret.firstOrderOptimality;
      auto delta_error = std::abs(norm_e - norm_e_prev);

      if(_params.verbosity == VerbosityType::kIteration) {
        printf(fmt_str, ret.numIterations, norm_e, norm_g, norm_dp, delta_error);
      }

      norm_e_prev = norm_e;
    }

    return ret;
  }

  AlgorithmParameters _params;
  LinearSystemBuilder _sys_builder;
  std::vector<float> _residuals;
  std::vector<uint8_t> _valid;
}; // PoseEstimator

struct VisualOdometry::Impl
{
  /** size of the image at the highest resolution */
  ImageSize _image_size;

  /** parameters */
  AlgorithmParameters _params;

  /** pyramid of template data */
  std::vector<UniquePointer<TemplateData>> _template_data_pyr;

  /** pose initialization in between keyframes */
  Matrix44 _T_init;

  std::vector<cv::Mat> _image_pyramid;
  KeyFrameCandidate _kf_candidate;

  std::vector<PoseEstimator> _pose_estimator_pyr;

  /**
   * \param K the intrinsic matrix at the highest resolution
   * \param b the stereo baseline at the highest resolution
   * \param params AlgorithmParameters
   */
  Impl(const Matrix33& KK, float b, ImageSize image_size, AlgorithmParameters params)
      : _image_size(image_size), _params(params), _T_init(Matrix44::Identity())
  {
    assert( _image_size.rows > 0 && _image_size.cols > 0 );

    // auto decide the number of levels of params.numPyramidLevels <= 0
    int num_levels = params.numPyramidLevels <= 0 ?
        getNumberOfPyramidLevels(std::min(_image_size.rows, _image_size.cols), 40) :
        params.numPyramidLevels;

    _params.numPyramidLevels = num_levels;
    dprintf("numPyramidLevels = %d\n", num_levels);

    AlgorithmParameters params_low_res = _params;

    // relatex the tolerance for low pyramid levels (for speed)
    if(params.relaxTolerancesForCoarseLevels) {
      params_low_res.parameterTolerance *= 10;
      params_low_res.functionTolerance *= 10;
      params_low_res.gradientTolerance *= 10;
      params_low_res.maxIterations = std::min(_params.maxIterations, 42);
    }

    // create the data per pyramid level
    Matrix33 K(KK);
    for(int i = 0; i < num_levels; ++i) {
      auto d = make_unique<TemplateData>(i != 0 ? params_low_res : params, K, b, i);
      _template_data_pyr.push_back(std::move(d));
      K *= 0.5; K(2,2) = 1.0; // K is cut by half
      b *= 2.0; // b *2, this was pose does not need rescaling across levels

      _pose_estimator_pyr.push_back(i != 0 ? params_low_res : params);
    }

    _image_pyramid.resize(num_levels);
  }

  ~Impl() {}

  /**
   * \param I pointer to the image
   * \param D poitner to the disparity
   */
  Result addFrame(const uint8_t*, const float*);

  void setAsKeyFrame(const cv::Mat&);

  void setImagePyramid(cv::Mat I)
  {
    assert(_image_pyramid.size() != 0);
    _image_pyramid[0] = I;
    for(size_t i = 1; i < _image_pyramid.size(); ++i)
      cv::pyrDown(_image_pyramid[i-1], _image_pyramid[i]);
  }

}; // Impl

VisualOdometry::VisualOdometry(const Matrix33& K, float baseline,
                               ImageSize image_size, AlgorithmParameters params)
    : _impl(new Impl(K, baseline, image_size, params)) {}

VisualOdometry::~VisualOdometry() { delete _impl; }

Result VisualOdometry::addFrame(const uint8_t* image, const float* disparity)
{
  return _impl->addFrame(image, disparity);
}

void VisualOdometry::Impl::setAsKeyFrame(const cv::Mat& D)
{
  assert( _image_pyramid.size() == _template_data_pyr.size() );

  for(size_t i = 0; i < _image_pyramid.size(); ++i) {
    _template_data_pyr[i]->compute(_image_pyramid[i], D);
    dprintf("[%dx%d] -> %d points\n", _image_pyramid[i].cols, _image_pyramid[i].rows,
            _template_data_pyr[i]->numPoints());
  }
}

Result VisualOdometry::Impl::addFrame(const uint8_t* image, const float* disparity)
{
  cv::Mat I, D;
  cv::Mat(_image_size.rows, _image_size.cols, CV_8UC1, (void*)image).copyTo(I);
  cv::Mat(_image_size.rows, _image_size.cols, CV_32FC1, (void*)disparity).copyTo(D);

  setImagePyramid(I);

  Result ret;
  ret.optimizerStatistics.resize(_params.numPyramidLevels);
  if(0 == _template_data_pyr.front()->numPoints()) {
    setAsKeyFrame(D);
    ret.isKeyFrame = true;
    ret.pose = _T_init;
    ret.covriance.setIdentity();

    return ret; // special case for the first frame
  }

  assert( _pose_estimator_pyr.size() == _template_data_pyr.size() &&
         _image_pyramid.size() == _pose_estimator_pyr.size() );

  ret.pose = _T_init;
  for(int i = _pose_estimator_pyr.size()-1; i >= 0; --i) {
    auto ss = _pose_estimator_pyr[i].run(_template_data_pyr[i].get(), _image_pyramid[i], ret.pose);
    ret.optimizerStatistics.push_back(ss);
  }


  return ret;
}

int VisualOdometry::numPointsAtLevel(int level) const
{
  return _impl->_template_data_pyr[level]->numPoints();
}

}; // bpvo

