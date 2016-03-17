#include "bpvo/vo_kf.h"
#include "bpvo/utils.h"
#include "bpvo/parallel_tasks.h"
#include "bpvo/dense_descriptor_pyramid.h"

#include <cmath>

namespace bpvo {

template <typename T> static inline
cv::Mat ToOpenCV(const T* p, const ImageSize& siz)
{
  return cv::Mat(siz.rows, siz.cols, cv::DataType<T>::type, (void*) p);
}


class VisualOdometryPoseEstimator
{
 public:
  /**
   * \param K the intrinsics matrix
   * \param b the stereo baseline
   * \param AlgorithmParameters parameters
   */
  VisualOdometryPoseEstimator(const Matrix33& K, const float baseline, AlgorithmParameters);

  /**
   * \param DenseDescriptorPyramid
   * \param D the disparity at the finest resolution
   */
  void setTemplate(DenseDescriptorPyramid&, const cv::Mat& D);

  /**
   * Estimates the pose wrt to the TemplateData
   *
   * \param DenseDescriptorPyramid of the input frame data
   * \param T_init pose initialization
   * \param T_est  the estimated pose
   *
   * NOTE: setTemplate must be called before calling this function
   */
  std::vector<OptimizerStatistics>
  estimatePose(DenseDescriptorPyramid&, const Matrix44& T_init, Matrix44& T_est);

  /**
   * \return true if template data was set
   */
  bool hasTemplate() const;

 private:
  AlgorithmParameters _params;
  PoseEstimatorGN<TemplateData> _pose_estimator;
  PoseEstimatorParameters _pose_est_params;
  PoseEstimatorParameters _pose_est_params_low_res;
  std::vector<UniquePointer<TemplateData>> _tdata_pyr;

 protected:
  OptimizerStatistics estimatePosetAtLevel(int level, DenseDescriptorPyramid&,
                                           const Matrix44& T_init, Matrix44& T_est);
}; // VisualOdometryPoseEstimator


VisualOdometryPoseEstimator::
VisualOdometryPoseEstimator(const Matrix33& K, const float b, AlgorithmParameters p)
  : _params(p)
  , _pose_estimator()
  , _pose_est_params(p)
  , _pose_est_params_low_res(p)
  , _tdata_pyr(p.numPyramidLevels)
{

  THROW_ERROR_IF(_tdata_pyr.empty(), "numPyramidLevels must be > 0");

  Matrix33 K_pyr(K);
  float b_pyr(b);
  _tdata_pyr.front() = make_unique<TemplateData>(0, K_pyr, b_pyr, p);
  for(size_t i = 1; i < _tdata_pyr.size(); ++i)
  {
    K_pyr *= 0.5; K_pyr(2,2) = 1.0; b_pyr *= 2.0;
    _tdata_pyr[i] = make_unique<TemplateData>(i, K_pyr, b_pyr, p);
  }
}

void VisualOdometryPoseEstimator::
setTemplate(DenseDescriptorPyramid& desc_pyr, const cv::Mat& D)
{
  ParallelTasks tasks(std::min(desc_pyr.size(), 4));
  for(int i = 0; i < desc_pyr.size(); ++i)
  {
    tasks.add([=,&desc_pyr]() {
          desc_pyr.compute(i);
          _tdata_pyr[i]->setData(desc_pyr[i], D);
        });
  }

  tasks.wait();
}

std::vector<OptimizerStatistics> VisualOdometryPoseEstimator::
estimatePose(DenseDescriptorPyramid& desc_pyr, const Matrix44& T_init, Matrix44& T_est)
{

  std::vector<OptimizerStatistics> ret(desc_pyr.size());
  _pose_estimator.setParameters(_pose_est_params_low_res);
  T_est = T_init;
  for(int i = desc_pyr.size()-1; i >= _params.maxTestLevel; --i)
  {
    ret[i] = estimatePosetAtLevel(i, desc_pyr, T_est, T_est);
  }

  return ret;
}

OptimizerStatistics VisualOdometryPoseEstimator::
estimatePosetAtLevel(int level, DenseDescriptorPyramid& desc_pyr,
                     const Matrix44& T_init, Matrix44& T_est)
{
  if(level >= _params.maxTestLevel)
    _pose_estimator.setParameters(_pose_est_params);

  desc_pyr.compute(level);
  T_est = T_init;
  return _pose_estimator.run(_tdata_pyr[level].get(), desc_pyr[level], T_est);
}

bool VisualOdometryPoseEstimator::hasTemplate() const
{
  return _tdata_pyr[_params.maxTestLevel]->numPoints() > 0;
}

struct VisualOdometryWithKeyFraming::KeyFrameCandidate
{
}; // KeyFrameCandidate

VisualOdometryWithKeyFraming::
VisualOdometryWithKeyFraming(const Matrix33& K, const float b,
                             ImageSize s, AlgorithmParameters p)
  : _params(p)
  , _image_size(s)
  , _T_kf(Matrix44::Identity())
  , _kf_candidate(make_unique<KeyFrameCandidate>())
{
  if(_params.numPyramidLevels <= 0)
  {
    _params.numPyramidLevels = 1 + std::round(
        std::log2(std::min(s.rows, s.cols) / (double) _params.minImageDimensionForPyramid));
  }

  _vo_pose = make_unique<VisualOdometryPoseEstimator>(K, b, _params);
}

VisualOdometryWithKeyFraming::~VisualOdometryWithKeyFraming() {}

Result VisualOdometryWithKeyFraming::
addFrame(const uint8_t* image_ptr, const float* disparity_ptr)
{
  const auto I = ToOpenCV(image_ptr, _image_size);
  const auto D = ToOpenCV(disparity_ptr, _image_size);

  if(!_desc_pyr)
  {
    _desc_pyr = make_unique<DenseDescriptorPyramid>(
        _params.descriptor, _params.numPyramidLevels, I, _params);

    Result ret;
    ret.pose.setIdentity();
    ret.covariance.setIdentity();
    ret.isKeyFrame = true;
    ret.keyFramingReason = KeyFramingReason::kFirstFrame;
    ret.optimizerStatistics.resize(_desc_pyr->size());
    for(auto& s : ret.optimizerStatistics) {
      s.numIterations = 0;
      s.finalError = 0.0f;
      s.firstOrderOptimality = 0.0f;
      s.status = PoseEstimationStatus::kFunctionTolReached;
    }

    return ret;
  }

  _desc_pyr->setImage(I);
  Matrix44 T_est;

  Result ret;
  ret.optimizerStatistics = _vo_pose->estimatePose(*_desc_pyr, _T_kf, T_est);
  _vo_pose->setTemplate(*_desc_pyr, D);

  return ret;
}

} // bpvo

