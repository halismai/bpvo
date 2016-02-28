#include <dmv/image_data.h>
#include <bpvo/imgproc.h>

namespace bpvo {
namespace dmv {

ImageData::ImageData(int id, const cv::Mat& I)
    : _frame_id(id)
{
  set(id, I);
}

const ImageData& ImageData::set(int id, const cv::Mat& I)
{
  _frame_id = id;

  _I = I.clone();

  imgradient(_I, _Ix, ImageGradientDirection::X);
  imgradient(_I, _Iy, ImageGradientDirection::Y);

  return *this;
}


} // dmv
} // bpvo
