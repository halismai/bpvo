#include "bpvo/utils.h"
#include "dmv/patch.h"
#include <opencv2/core/core.hpp>

namespace bpvo {
namespace dmv {

void Patch3x3::set(const cv::Mat& I, const ImagePoint& p)
{
  int stride = I.step / I.elemSize1();
  int radius = 1;

  THROW_ERROR_IF( p.y() < radius || p.y() > I.rows - radius - 1 ||
                  p.x() < radius || p.x() > I.cols - radius - 1,
                  Format_("point is outside image bounds [%g,%g]", p.x(), p.y()));

  extractPatch(I.ptr<uint8_t>(), stride, p.y(), p.x(), radius, _data);
}

} // dmv
} // bpvo

