#include "bpvo/channels.h"
#include "bpvo/census.h"
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>

namespace bpvo {

RawIntensity::RawIntensity() {}

RawIntensity::RawIntensity(const cv::Mat& I)
{
  this->compute(I);
}

void RawIntensity::compute(const cv::Mat& I)
{
  assert( I.type() == cv::DataType<uint8_t>::type );
  _I = I.clone();
}

cv::Mat_<float> RawIntensity::computeSaliencyMap() const
{
  int rows = _I.rows,
      cols = _I.cols;

  cv::Mat_<float> ret(rows, cols);
  auto dst_ptr = ret.ptr<float>();
  memset(dst_ptr, 0.0f, sizeof(float)*cols);
  dst_ptr += cols;

  for(int y = 1; y < rows - 1; ++y) {
    auto s0 = _I.ptr<uint8_t>(y - 1),
         s1 = _I.ptr<uint8_t>(y + 1),
         s = _I.ptr<uint8_t>(y);

    dst_ptr[0] = 0.0f;
#pragma omp simd
    for(int x = 1; x < cols - 1; ++x) {
      dst_ptr[x] = std::fabs(static_cast<float>(s[x+1]) - static_cast<float>(s[x-1])) +
          std::fabs(static_cast<float>(s1[x]) - static_cast<float>(s0[x]));
    }

    dst_ptr[cols - 1] = 0.0f;
    dst_ptr += cols;
  }

  memset(dst_ptr, 0, sizeof(float)*cols);

  return ret;
}

template <typename DstType> static inline
void extractChannel(const cv::Mat& C, cv::Mat& dst, int b, float sigma)
{
  assert( b >= 0 && b < 8  );

  dst.create(C.size(), cv::DataType<DstType>::type);

  auto src_ptr = C.ptr<const uint8_t>();
  auto dst_ptr = dst.ptr<DstType>();
  auto n = C.rows * C.cols;

#pragma omp simd
  for(int i = 0; i < n; ++i) {
    dst_ptr[i] = static_cast<DstType>( (src_ptr[i] & (1<<b)) >> b );
  }

  if(sigma > 0.0f)
    cv::GaussianBlur(dst, dst, cv::Size(5,5), sigma, sigma);
}

void BitPlanes::compute(const cv::Mat& I)
{
  assert( I.type() == cv::DataType<uint8_t>::type );

  auto C = census(I, _sigma_ct);
  for(size_t i = 0; i < NumChannels; ++i) {
    extractChannel<float>(C, _channels[i], i, _sigma_bp);
  }
}

cv::Mat_<float> BitPlanes::computeSaliencyMap() const
{
  assert( !_channels.front().empty() );

  auto rows = _channels[0].rows, cols = _channels[0].cols;

  // TODO there is a lot of duplicate code here fix

  cv::Mat_<float> ret(rows, cols);
  {
    //
    // first channel, we initialize the map
    //
    auto dst_ptr = ret.ptr<float>();
    memset(dst_ptr, 0.0f, sizeof(float)*cols);
    dst_ptr += cols;
    for(int y = 1; y < rows - 1; ++y) {
      auto s0 = _channels[0].ptr<uint8_t>(y - 1),
           s1 = _channels[0].ptr<uint8_t>(y + 1),
           s = _channels[0].ptr<uint8_t>(y);

      dst_ptr[0] = 0.0f;
#pragma omp simd
      for(int x = 1; x < cols - 1; ++x) {
        dst_ptr[x] = std::fabs(static_cast<float>(s[x+1]) - static_cast<float>(s[x-1])) +
            std::fabs(static_cast<float>(s1[x]) - static_cast<float>(s0[x]));
      }

      dst_ptr[cols - 1] = 0.0f;
      dst_ptr += cols;
    }
    memset(dst_ptr, 0, sizeof(float)*cols);
  }

  {
    // for the rest of the channels, we +=
    for(int c = 1; c < NumChannels; ++c) {
      auto dst_ptr = ret.ptr<float>();
      memset(dst_ptr, 0.0f, sizeof(float)*cols);
      dst_ptr += cols;
      for(int y = 1; y < rows - 1; ++y) {
        auto s0 = _channels[c].ptr<uint8_t>(y - 1),
             s1 = _channels[c].ptr<uint8_t>(y + 1),
             s = _channels[c].ptr<uint8_t>(y);

        dst_ptr[0] = 0.0f;
#pragma omp simd
        for(int x = 1; x < cols - 1; ++x) {
          dst_ptr[x] += std::fabs(static_cast<float>(s[x+1]) - static_cast<float>(s[x-1])) +
              std::fabs(static_cast<float>(s1[x]) - static_cast<float>(s0[x]));
        }

        dst_ptr[cols - 1] = 0.0f;
        dst_ptr += cols;
      }
      memset(dst_ptr, 0, sizeof(float)*cols);
    }
  }

  return ret;
}

}; // bpvo

