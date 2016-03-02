#include "dmv/photo_vo_patch.h"
#include "dmv/patch_util.h"

#include "bpvo/utils.h"

#include <opencv2/core/core.hpp>

namespace bpvo {
namespace dmv {

PhotoVoPatch::PhotoVoPatch(const Mat_<double,3,3>& K, double b, Config conf)
    : PhotoVoBase(K, b, conf) {}

PhotoVoPatch::~PhotoVoPatch() {}

void PhotoVoPatch::setImageData(const cv::Mat& /*image*/, const cv::Mat& /* D*/)
{
  THROW_ERROR_IF( this->_points.empty(), "there are no points" );
}

} // dmv
} // bpvo
