#include "bpvo/vo.h"
#include <opencv2/core/core.hpp>

#include <memory>

namespace bpvo {

struct VisualOdometry::Impl
{
  Impl(ImageSize image_size, AlgorithmParameters params)
      : _image_size(image_size), _params(params)
  {
    assert( _image_size.rows > 0 && _image_size.cols > 0 );
  }

  ImageSize _image_size;
  AlgorithmParameters _params;

  Result addFrame(const uint8_t*, const float*);
}; // Impl

VisualOdometry::VisualOdometry(ImageSize image_size, AlgorithmParameters params)
    : _impl(new Impl(image_size, params)) {}

VisualOdometry::~VisualOdometry() { delete _impl; }

Result VisualOdometry::addFrame(const uint8_t* image, const float* disparity)
{
  return _impl->addFrame(image, disparity);
}

}; // bpvo

