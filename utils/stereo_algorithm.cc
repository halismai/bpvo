#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "bpvo/config_file.h"
#include "utils/stereo_algorithm.h"

namespace bpvo {

struct StereoAlgorithm::Impl
{
  enum class Algorithm
  {
    BlockMatching,
    SemiGlobalBlockMatching
  }; // Algorithm

  Impl(const ConfigFile& cf)
      : _algorithm(Algorithm::BlockMatching), _state(NULL)
  {
    auto alg = cf.get<std::string>("StereoAlgorithm", "BlockMatching");

    if(icompare("SGBM", alg) || icompare("SemiGlobalBlockMatching", alg))
    {
      _algorithm = Algorithm::SemiGlobalBlockMatching;
      _sgbm = make_unique<cv::StereoSGBM>(
          cf.get<int>("minDisparity"),
          cf.get<int>("numberOfDisparities"),
          cf.get<int>("SADWindowSize", 3),
          cf.get<int>("P1", 0),
          cf.get<int>("P2", 0),
          cf.get<int>("uniquenessRatio", 0),
          cf.get<int>("speckleWindowSize", 0),
          cf.get<int>("speckleRange", 0),
          (bool) cf.get<int>("fullDP", 0));
    } else
    {
      _algorithm = Algorithm::BlockMatching;
      _state = cvCreateStereoBMState();
      // default values from opencv/modules/calib3d/src/stereobm.cpp
      _state->preFilterType = cf.get<int>("preFilterType", CV_STEREO_BM_XSOBEL);
      _state->preFilterSize = cf.get<int>("preFilterSize", 9);
      _state->preFilterCap = cf.get<int>("preFilterCap", 31);

      _state->SADWindowSize = cf.get<int>("SADWindowSize", 15);
      _state->minDisparity = cf.get<int>("minDisparity", 0);
      _state->numberOfDisparities = cf.get<int>("numberOfDisparities"); // must be provided

      _state->textureThreshold = cf.get<int>("textureThreshold", 10);
      _state->uniquenessRatio = cf.get<int>("uniquenessRatio", 15);
      _state->speckleWindowSize = cf.get<int>("speckleWindowSize", 0);
      _state->speckleRange = cf.get<int>("speckleRange", 0);
      _state->trySmallerWindows = cf.get<int>("trySmallerWindows", 0);
      _state->disp12MaxDiff = cf.get<int>("disp12MaxDiff", -1);
    }
  }

  ~Impl()
  {
    if(_state) cvReleaseStereoBMState(&_state);
  }

  inline void run(const cv::Mat& left, const cv::Mat& right, cv::Mat& dmap)
  {
    _dmap_buffer.create(left.size(), CV_16SC1);

    if(_state)
    {
      const CvMat left_ = left;
      const CvMat right_ = right;
      CvMat dmap_ = _dmap_buffer;

      cvFindStereoCorrespondenceBM(&left_, &right_, &dmap_, _state);
    } else
    {
      _sgbm->operator()(left, right, _dmap_buffer);
    }

    // convert to float
    _dmap_buffer.convertTo(dmap, CV_32FC1, 1.0 / 16.0, 0.0 );
  }


  inline short getInvalidValue() const
  {
    return static_cast<short>( _state->minDisparity - 1 );
  }

  inline float getInvalidValueFloat() const
  {
    return static_cast<float>(getInvalidValue()) / 16.0f;
  }

  Algorithm _algorithm;
  CvStereoBMState* _state;
  UniquePointer<cv::StereoSGBM> _sgbm;
  cv::Mat _dmap_buffer;
}; // impl

StereoAlgorithm::StereoAlgorithm(const ConfigFile& cf)
  : _impl(make_unique<Impl>(cf)) {}

StereoAlgorithm::StereoAlgorithm(std::string conf_fn)
  : StereoAlgorithm(ConfigFile(conf_fn)) {}

  StereoAlgorithm::~StereoAlgorithm() {}

void StereoAlgorithm::run(const cv::Mat& left, const cv::Mat& right, cv::Mat& dmap)
{
  _impl->run(left, right, dmap);
}

float StereoAlgorithm::getInvalidValue() const { return _impl->getInvalidValueFloat(); }

} // bpvo
