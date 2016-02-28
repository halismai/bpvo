#ifndef BPVO_DMV_IMAGE_DATA_H
#define BPVO_DMV_IMAGE_DATA_H

#include <opencv2/core/core.hpp>


namespace bpvo {
namespace dmv {

class ImageData
{
 public:
  inline ImageData() : _frame_id(-1) {}
  explicit ImageData(int frame_id, const cv::Mat& I);

  const ImageData& set(int frame_id, const cv::Mat& I);

  inline int id() const { return _frame_id; }

 protected:
  int _frame_id;
  cv::Mat _I, _Ix, _Iy;
}; // ImageData

}; // dmv
}; // bpvo

#endif // BPVO_DMV_IMAGE_DATA_H
