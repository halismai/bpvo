#include "dmv/patch.h"
#include <opencv2/core/core.hpp>

namespace bpvo {
namespace dmv {

void Patch3x3::set(const cv::Mat& I, const ImagePoint& p)
{
  int stride = I.step / I.elemSize1();
  int radius = 1;

  extractPatch(I.ptr<uint8_t>(), stride, p.y(), p.x(), radius, _data);
}

} // dmv
} // bpvo

