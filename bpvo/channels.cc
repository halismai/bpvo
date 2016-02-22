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

#include "bpvo/channels.h"
#include "bpvo/census.h"
#include "bpvo/imgproc.h"
#include "bpvo/parallel.h"
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>

#define DO_BITPLANES_WITH_TBB 1
#if DO_BITPLANES_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif


namespace bpvo {

RawIntensity::RawIntensity(float,float) {}

RawIntensity::RawIntensity(const cv::Mat& I)
{
  this->compute(I);
}

void RawIntensity::compute(const cv::Mat& I)
{
  I.convertTo(_I, CV_32FC1);
}

void RawIntensity::computeSaliencyMap(cv::Mat_<float>& buffer) const
{
  gradientAbsoluteMagnitude(_I, buffer);
}

template <typename DstType> static inline
void extractChannel(const cv::Mat& C, cv::Mat& dst, int b, float sigma)
{
  assert( b >= 0 && b < 8  );

  dst.create(C.size(), cv::DataType<DstType>::type);

  auto src_ptr = C.ptr<const uint8_t>();
  auto dst_ptr = dst.ptr<DstType>();
  auto n = C.rows * C.cols;

#if defined(WITH_OPENMP)
#pragma omp simd
#endif
  for(int i = 0; i < n; ++i) {
    dst_ptr[i] = static_cast<DstType>( (src_ptr[i] & (1<<b)) >> b );
  }

  if(sigma > 0.0f)
    cv::GaussianBlur(dst, dst, cv::Size(5,5), sigma, sigma);
}

namespace {

struct BitPlanesComputeBody : public ParallelForBody
{
 public:
  typedef typename BitPlanes::ChannelsArray ChannelsArray;

 public:
  BitPlanesComputeBody(const cv::Mat& I, float sigma_ct, float sigma_bp,
                       ChannelsArray& channels)
      : _C(census(I, sigma_ct)), _sigma_bp(sigma_bp), _channels(channels) {}

  inline void operator()(const Range& range) const
  {
    for(int c = range.begin(); c != range.end(); ++c)
    {
      extractChannel<float>(_C, _channels[c], c, _sigma_bp);
    }
  }

 protected:
  cv::Mat _C;
  float _sigma_bp;
  ChannelsArray& _channels;
};

}; // namespace

void BitPlanes::compute(const cv::Mat& I)
{
  assert( I.type() == cv::DataType<uint8_t>::type );

  BitPlanesComputeBody func(I, _sigma_ct, _sigma_bp, _channels);
  parallel_for(Range(0, NumChannels), func);
}

void BitPlanes::computeSaliencyMap(cv::Mat_<float>& dst) const
{
  assert( !_channels.front().empty() );

  dst.create(_channels[0].size());
  gradientAbsoluteMagnitude(_channels[0], dst);
  for(int i = 1; i < NumChannels; ++i)
    gradientAbsoluteMagnitudeAcc(_channels[i], dst.ptr<float>());
}

}; // bpvo

