/*
   This file is part of bpvo.

   bpvo is free software: you can redistribute it and/or modify
   it under the terms of the Lesser GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   bpvo is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   Lesser GNU General Public License for more details.

   You should have received a copy of the Lesser GNU General Public License
   along with bpvo.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Contributor: halismai@cs.cmu.edu
 */

#include "bpvo/census.h"
#include "bpvo/v128.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace bpvo {

static const __m128i K0x01 = _mm_set1_epi8(0x01);
static const __m128i K0x02 = _mm_set1_epi8(0x02);
static const __m128i K0x04 = _mm_set1_epi8(0x04);
static const __m128i K0x08 = _mm_set1_epi8(0x08);
static const __m128i K0x10 = _mm_set1_epi8(0x10);
static const __m128i K0x20 = _mm_set1_epi8(0x20);
static const __m128i K0x40 = _mm_set1_epi8(0x40);
static const __m128i K0x80 = _mm_set1_epi8(0x80);

/**
 * computes the Census Transform for 16 pixels at once
 */
static FORCE_INLINE void censusOp(const uint8_t* src, int stride, uint8_t* dst)
{

#define C_OP >=
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
#undef C_OP
}

cv::Mat census(const cv::Mat& src, float s)
{
  assert( src.type() == CV_8UC1 && src.channels() == 1 && src.isContinuous() );

  cv::Mat image = src;
  if(s > 0.0f)
    cv::GaussianBlur(src, image, cv::Size(3,3), s, s);

  cv::Mat dst(src.size(), CV_8UC1);
  const int W = 1 + ((src.cols - 2) & ~15);
  auto src_ptr = image.ptr<const uint8_t>();
  auto dst_ptr = dst.ptr<uint8_t>();

  memset(dst_ptr, 0, src.cols); // set two rows to 0
  src_ptr += src.cols;
  dst_ptr += dst.cols;

  for(int r = 2; r < src.rows; ++r, src_ptr += src.cols, dst_ptr += dst.cols)
  {
    *(dst_ptr + 0) = 0;
    for(int c = 1; c < W; c += 16)
      censusOp(src_ptr + c, src.cols, dst_ptr + c);

    if(W != src.cols - 1)
      censusOp(src_ptr + src.cols - 1 - 16, src.cols, dst_ptr + dst.cols - 1 - 16);

    *(dst_ptr + dst.cols - 1) = 0;
  }

  memset(dst_ptr, 0, dst.cols);

  return dst;
}

} // bpvo


