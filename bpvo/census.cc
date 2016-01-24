#include "bpvo/census.h"
#include "bpvo/v128.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace bpvo {

static const __m128i K0x01 = _mm_set1_epi8(0x01);
static const __m128i K0x02 = _mm_set1_epi8(0x02);
static const __m128i K0x04 = _mm_set1_epi8(0x04);
static const __m128i K0x08 = _mm_set1_epi8(0x08);
static const __m128i K0x10 = _mm_set1_epi8(0x10);
static const __m128i K0x20 = _mm_set1_epi8(0x20);
static const __m128i K0x40 = _mm_set1_epi8(0x40);
static const __m128i K0x80 = _mm_set1_epi8(0x80);

#define C_OP >=
/**
 * computes the Census Transform for 16 pixels at once
 */
static inline void censusOp(const uint8_t* src, int stride, uint8_t* dst)
{
  const v128 c(src);
  _mm_storeu_si128((__m128i*) dst,
                   ((v128(src - stride - 1) C_OP c) & K0x01) |
                   ((v128(src - stride    ) C_OP c) & K0x02) |
                   ((v128(src - stride + 1) C_OP c) & K0x04) |
                   ((v128(src          - 1) C_OP c) & K0x08) |
                   ((v128(src          + 1) C_OP c) & K0x10) |
                   ((v128(src + stride - 1) C_OP c) & K0x20) |
                   ((v128(src + stride    ) C_OP c) & K0x40) |
                   ((v128(src + stride + 1) C_OP c) & K0x80));
}
#undef C_OP

cv::Mat census(const cv::Mat& src, float s)
{
  assert( src.type() == CV_8UC1 && src.channels() == 1 && src.isContinuous() );
  cv::Mat image;
  if(s > 0.0f)
    cv::GaussianBlur(src, image, cv::Size(), s, s);
  else
    image = src;

  cv::Mat dst(src.size(), CV_8UC1);
  const int W = 1 + ((src.cols - 2) & ~15);
  auto src_ptr = image.ptr<const uint8_t>();
  auto dst_ptr = dst.ptr<uint8_t>();

  memset(dst_ptr, 0, src.cols);
  src_ptr += src.cols;
  dst_ptr += dst.cols;

  for(int r = 2; r < src.rows; ++r, src_ptr += src.cols, dst_ptr += dst.cols)
  {
    *(dst_ptr  + 0) = 0;
    for(int c = 0; c < W; c += 16)
      censusOp(src_ptr + c, src.cols, dst_ptr + c);

    if(W != src.cols - 1)
      censusOp(src_ptr + src.cols - 1 - 16, src.cols, dst_ptr + dst.cols - 1 - 16);

    *(dst_ptr + dst.cols - 1) = 0;
  }

  memset(dst_ptr, 0, dst.cols);

  return dst;
}

} // bpvo


