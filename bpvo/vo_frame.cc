#include "bpvo/vo_frame.h"
#include "bpvo/template_data.h"
#include "bpvo/dense_descriptor.h"
#include "bpvo/dense_descriptor_pyramid.h"

#include <opencv2/core/core.hpp>

namespace bpvo {

VisualOdometryFrame::VisualOdometryFrame(const Matrix33& K, float b, const AlgorithmParameters& p)
    : _max_test_level(p.maxTestLevel)
    , _has_data(false)
    , _disparity(make_unique<cv::Mat>())
    , _desc_pyr(make_unique<DenseDescriptorPyramid>(p))
    , _set_template_thread(&VisualOdometryFrame::set_template_data, this)
    , _is_tdata_ready(false)
    , _should_quit(false)
{
  Matrix33 K_pyr(K);
  float b_pyr(b);

  _tdata_pyr.push_back(make_unique<TemplateData>(0, K_pyr, b_pyr, p));
  for(int i = 1; i < p.numPyramidLevels; ++i)
  {
    K_pyr *= 0.5; K_pyr(2,2) = 1.0f; b_pyr *= 2.0;
    _tdata_pyr.push_back( make_unique<TemplateData>(i, K_pyr, b_pyr, p) );
  }
}

VisualOdometryFrame::~VisualOdometryFrame()
{
  _should_quit = true;

  _has_template_data.notify_one();

  if(_set_template_thread.joinable()) {
    _set_template_thread.join();
  }
}

const DenseDescriptor* VisualOdometryFrame::getDenseDescriptorAtLevel(size_t l) const
{
  assert( l  < _desc_pyr->size() );
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
  disparity.copyTo( *_disparity );
  _desc_pyr->init(image);

  _has_data = true;
  _is_tdata_ready = false;

  std::lock_guard<std::mutex> lock(_mutex);
  _has_template_data.notify_one();
}

void VisualOdometryFrame::set_template_data()
{
  while(!_should_quit)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _has_template_data.wait(lock);

    if(!_should_quit && !_disparity->empty())
    {
      for(size_t i = 0; i < _tdata_pyr.size(); ++i)
        _tdata_pyr[i]->setData( _desc_pyr->operator[](i), *_disparity);

      _is_tdata_ready = true;
    }

    lock.unlock();
  }
}

}; // bpvo
