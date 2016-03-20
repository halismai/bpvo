#include "bpvo/dense_descriptor_pyramid_level.h"
#include "bpvo/dense_descriptor.h"
#include "bpvo/utils.h"

#include <opencv2/core/core.hpp>

namespace bpvo {

struct DenseDescriptorPyramidLevel::Impl
{
  Impl(const AlgorithmParameters& p, int l)
   : _desc(UniquePointer<DenseDescriptor>(DenseDescriptor::Create(p, l))) {}

  void setImage(const cv::Mat& image, bool compute_now)
  {
    _image = image;
    if(compute_now)
    {
      _desc->compute( image );
      _has_data = true;
    }
  }


  const cv::Mat& getDescriptorData(bool force_recompute)
  {
    THROW_ERROR_IF(_image.empty(), "must set the image before calling getDescriptorData\n");

    if(force_recompute || !_has_data)
    {
      _desc->compute(_image);
    }

    _has_data = true;
    return _image;
  }

  UniquePointer<DenseDescriptor> _desc;

  cv::Mat _image;
  bool _has_data = false;
}; // DenseDescriptorPyramidLevel::Impl

DenseDescriptorPyramidLevel::
DenseDescriptorPyramidLevel(const AlgorithmParameters& p, int l)
  : _impl(make_unique<Impl>(p, l)) {}

DenseDescriptorPyramidLevel::
DenseDescriptorPyramidLevel(DenseDescriptorPyramidLevel&& other)
  : _impl(std::move(other._impl)) {}

DenseDescriptorPyramidLevel&
DenseDescriptorPyramidLevel::operator=(DenseDescriptorPyramidLevel&& other)
{
  if(this != &other)
  {
    _impl = std::move(other._impl);
  }

  return *this;
}

}; // bpvo
