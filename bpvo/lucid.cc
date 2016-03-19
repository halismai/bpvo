#include "bpvo/lucid.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <vector>

namespace bpvo {


template <typename T> static inline
std::vector<int> sort_perm(const std::vector<T>& v)
{
  std::vector<int> ret(v.size());
  std::iota(ret.begin(), ret.end(), 0);
  std::sort(ret.begin(), ret.end(), [&](int i1, int i2) { return v[i1] < v[i2]; });
  return ret;
}

LucidDescriptor::LucidDescriptor(int r, int b)
    : _radius(r), _blur_radius(b) {}

void LucidDescriptor::compute(const cv::Mat& src_, cv::Mat& dst) const
{
  int k = (_blur_radius*2 + 1) * (_blur_radius*2 + 1),
      k2 = (_radius*2 + 1) * (_radius*2 + 1);

  typedef int DstType;
  dst.create(k2, src_.rows*src_.cols, cv::DataType<DstType>::type);
  memset(dst.data, 0, sizeof(DstType) * _radius * dst.cols);

  cv::Mat src;
  cv::blur(src_, src, cv::Size(k,k));

  for(int y = _radius; y < src.rows - _radius - 1; ++y) {
    std::vector<uint8_t> tmp(k2);
    for(int x = _radius; x < src.cols - _radius - 1; ++x)
    {
      for(int r = -_radius, i=0; r <= _radius; ++r)
        for(int c = -_radius; c <= _radius; ++c, ++i)
          tmp[i] = src.at<uint8_t>(y+r, x+c);

      auto inds = sort_perm(tmp);
      // XXX
      memcpy(dst.data + y*src_.cols+x, inds.data(), sizeof(int)*inds.size());
    }

  }


}

} // bpvo
