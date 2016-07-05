#include "bpvo/vo_frame.h"
#include "bpvo/template_data.h"
#include "bpvo/dense_descriptor.h"
#include "bpvo/dense_descriptor_pyramid.h"
#include "bpvo/parallel_tasks.h"
#include "bpvo/utils.h"
#include "bpvo/image_pyramid.h"

#include <opencv2/core/core.hpp>

namespace bpvo {

VisualOdometryFrame::VisualOdometryFrame(const Matrix33& K, float b, const AlgorithmParameters& p)
    : _max_test_level(p.maxTestLevel)
    , _has_data(false)
    , _has_template(false)
    , _image(make_unique<cv::Mat>())
    , _disparity(make_unique<cv::Mat>())
    , _desc_pyr(make_unique<DenseDescriptorPyramid>(p))
{
  Matrix33 K_pyr(K);
  float b_pyr(b);

  _tdata_pyr.push_back(make_unique<TemplateData>(0, K_pyr, b_pyr, p));
  for(int i = 1; i < p.numPyramidLevels; ++i) {
    K_pyr *= 0.5; K_pyr(2,2) = 1.0f; b_pyr *= 2.0;
    _tdata_pyr.push_back( make_unique<TemplateData>(i, K_pyr, b_pyr, p) );
  }
}

VisualOdometryFrame::~VisualOdometryFrame() {}

const DenseDescriptor* VisualOdometryFrame::getDenseDescriptorAtLevel(size_t l) const
{
  assert( l < (size_t) _desc_pyr->size() );
  return _desc_pyr->operator[](l);
}

const TemplateData* VisualOdometryFrame::getTemplateDataAtLevel(size_t l) const
{
  assert( l < _tdata_pyr.size() );

  return _tdata_pyr[l].get();
}

int VisualOdometryFrame::numLevels() const { return _desc_pyr->size(); }

void VisualOdometryFrame::setData(const cv::Mat& image, const cv::Mat& disparity)
{
  image.copyTo( *_image );
  disparity.copyTo( *_disparity );
  _desc_pyr->init(image);

  _has_data = true;
}

const cv::Mat* VisualOdometryFrame::imagePointer() const { return _image.get(); }

const cv::Mat* VisualOdometryFrame::disparityPointer() const { return _disparity.get(); }

void VisualOdometryFrame::setTemplate()
{
  THROW_ERROR_IF(!_has_data, "no data in frame");

#define VO_FRAME_USE_PARALLEL 0

#if VO_FRAME_USE_PARALLEL
  ParallelTasks tasks(std::min(numLevels() - _max_test_level, 4));
#endif

  const cv::Mat& D = *_disparity;
  for(int i = _tdata_pyr.size()-1; i >= _max_test_level; --i)
  {
    auto code = [=]()
    {
      _tdata_pyr[i]->setData(_desc_pyr->operator[](i), D);
    }; // code

#if VO_FRAME_USE_PARALLEL
    tasks.add( code );
#else
    code();
#endif
  }

#if VO_FRAME_USE_PARALLEL
  tasks.wait();
#endif

#undef VO_FRAME_USE_PARALLEL

  _has_template = true;
}

void VisualOdometryFrame::setDataAndTemplate(const cv::Mat& image, const cv::Mat& disparity)
{
  ImagePyramid image_pyramid(numLevels());
  image_pyramid.compute(image);

#define VO_FRAME_USE_PARALLEL 1

#if VO_FRAME_USE_PARALLEL
  ParallelTasks tasks(std::min((int)_tdata_pyr.size(), 4));
#endif

  for(size_t i = 0; i < _tdata_pyr.size(); ++i)
  {
    auto code = [=]()
    {
      auto* desc = _desc_pyr->operator[](i);
      desc->compute(image_pyramid[i]);
      _tdata_pyr[i]->setData( desc, disparity );
    };

#if VO_FRAME_USE_PARALLEL
    tasks.add( code );
#else
    code();
#endif
  }

#if VO_FRAME_USE_PARALLEL
  tasks.wait();
#endif

#undef VO_FRAME_USE_PARALLEL
}

}; // bpvo

