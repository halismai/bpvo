#include "bpvo/vo_impl.h"

namespace bpvo {

VisualOdometry::VisualOdometry(const Matrix33& K, float baseline,
                               ImageSize image_size, AlgorithmParameters params)
    : _impl(new Impl(K, baseline, image_size, params)) {}

VisualOdometry::~VisualOdometry() { delete _impl; }

Result VisualOdometry::addFrame(const uint8_t* image, const float* disparity)
{
  return _impl->addFrame(image, disparity);
}

int VisualOdometry::numPointsAtLevel(int level) const
{
  return _impl->numPointsAtLevel(level);
}

auto VisualOdometry::pointsAtLevel(int level) const -> const PointVector&
{
  return _impl->pointsAtLevel(level);
}

}; // bpvo

