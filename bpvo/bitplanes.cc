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

#include "bpvo/bitplanes.h"
#include "bpvo/debug.h"
#include "bpvo/census.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstring>
#include <iostream>
#include <type_traits>

#include <emmintrin.h>

/**
 * extract the channels using a single core
 */
#define EXTRACT_CHANNELS_SERIAL 0
#if EXTRACT_CHANNELS_SERIAL
#define TBB_PREVIEW_SERIAL_SUBSET 1
#endif

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <Eigen/Core>

namespace bpvo {

static void GaussianBlur(cv::Mat& I, cv::Size ks, float s)
{
  if(s > 0.0f)
    cv::GaussianBlur(I, I, ks, s, s);
}


/**
 * Extracts the b-th  channel from the census transformed image and converts
 * the bit 'DstType'
 */
template <typename DstType>
void extractChannel(const cv::Mat& C, cv::Mat& BP, int b)
{
  BP.create(C.size(), cv::DataType<DstType>::type);
  auto src = C.ptr<const uint8_t>();
  auto dst = BP.ptr<DstType>();
  auto n = C.rows * C.cols;

#pragma omp simd
  for(int i = 0; i < n; ++i) {
    dst[i] = static_cast<DstType>( (src[i] & (1 << b)) >> b );
  }
}

/**
 * Make a channel of bitplanes
 *
 * First, we extract the bit from the census transform, then we optionally
 * smooth the image
 */
struct BitPlanesChannelMaker
{
  /**
   * \param C the census transform image
   * \param sigma smoothing to apply after extracting the bit
   * \param data output data
   */
  BitPlanesChannelMaker(cv::Mat C, float sigma, BitPlanesData& data)
      : _C(C), _sigma(sigma), _data(data) {}

  void operator()(const tbb::blocked_range<int>& range) const
  {
    for(int i = range.begin(); i != range.end(); ++i) {
      extractChannel<float>(_C, _data._channels[i], i);
      GaussianBlur(_data._channels[i], cv::Size(5,5), _sigma);
    }
  }

  const cv::Mat& _C;
  float _sigma;
  BitPlanesData& _data;
}; // BitPlanesChannelMaker


BitPlanesData::BitPlanesData(float s1, float s2)
  : _sigma_ct(s1), _sigma_bp(s2) {}

BitPlanesData::BitPlanesData(float s1, float s2, const cv::Mat& image)
  : BitPlanesData(s1, s2)
{
  computeChannels(image);
}

BitPlanesData& BitPlanesData::computeChannels(const cv::Mat& image)
{
  tbb::parallel_for(
      tbb::blocked_range<int>(0,8),
      BitPlanesChannelMaker(census(image, _sigma_ct), _sigma_bp, *this));

  return *this;
}

template <typename TSrc> static FORCE_INLINE
cv::Mat computeGradientAbsMagChannel(const cv::Mat& src)
{

  int rows = src.rows, cols = src.cols;
  cv::Mat dst(rows, cols, cv::DataType<TSrc>::type);

  auto dst_ptr = dst.ptr<TSrc>();
  auto src_ptr = src.ptr<const TSrc>();

  for(int y = 1; y < rows - 2; ++y) {
#pragma omp simd
    for(int x = 1; x < cols - 2; ++x) {
      int i = y*cols + x;
      dst_ptr[i] = fabs(src_ptr[i+1] - src_ptr[i-1]) + fabs(src_ptr[i+cols] - src_ptr[i-cols]);
    }
  }

  return dst;
}

template <typename TSrc> static FORCE_INLINE
void accumulateGradientAbsMag(const cv::Mat& src, cv::Mat& dst)
{
  int rows = src.rows, cols = src.cols;
  assert( rows == dst.rows && cols == dst.cols );

  auto dst_ptr = dst.ptr<TSrc>();
  auto src_ptr = src.ptr<const TSrc>();

  for(int y = 1; y < rows - 2; ++y) {
#pragma omp simd
    for(int x = 1; x < cols - 2; ++x) {
      int i = y*cols + x;
      dst_ptr[i] += fabs(src_ptr[i+1] - src_ptr[i-1]) +
          fabs(src_ptr[i+cols] - src_ptr[i-cols]);
    }
  }
}

void BitPlanesData::computeGradientAbsMag()
{
  assert( !_channels.front().empty() );

  int rows = _channels.front().rows, cols = _channels.front().cols;
  _gmag.create(rows, cols, CV_32FC1);


  // first loop we compute the gradientAbsMag for the first channnel.
  // In the second loop we do +=

  memset(_gmag.ptr<float>(), 0.0, sizeof(float)*cols);

  for(int y = 1; y < rows - 2; ++y) {
    auto s0 = _channels[0].ptr<const float>(y-1),
         s1 = _channels[0].ptr<const float>(y+1),
         s = _channels[0].ptr<const float>(y);

    auto d = _gmag.ptr<float>(y);
#pragma omp simd
    for(int x = 1; x < cols - 2; ++x) {
      d[x] = std::fabs(s[x+1] - s[x-1]) + std::fabs(s1[x] - s0[x]);
    }

    d[cols - 1] = 0.0f;
  }

  memset(_gmag.ptr<float>(rows-1), 0, sizeof(float)*cols);

  for(int cn = 1; cn < 8; ++cn) {
    for(int y = 1; y < rows - 2; ++y) {
      auto s0 = _channels[0].ptr<const float>(y-1),
           s1 = _channels[0].ptr<const float>(y+1),
           s = _channels[0].ptr<const float>(y);

      auto d = _gmag.ptr<float>(y);

#pragma omp simd
      for(int x = 1; x < cols - 2; ++x) {
        d[x] += std::fabs(s[x+1] - s[x-1]) + std::fabs(s1[x] - s0[x]);
      }
    }
  }
}

}; // bpvo

#if EXTRACT_CHANNELS_SERIAL
#undef TBB_PREVIEW_SERIAL_SUBSET
#endif

#undef EXTRACT_CHANNELS_SERIAL

