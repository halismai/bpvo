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

void BitPlanes::compute(const cv::Mat& I)
{
  assert( I.type() == cv::DataType<uint8_t>::type );

  auto C = census(I, _sigma_ct);

#if DO_BITPLANES_WITH_TBB
  tbb::parallel_for(tbb::blocked_range<int>(0, NumChannels),
                    [=](const tbb::blocked_range<int>& r)
                    {
                      for(int c = r.begin(); c != r.end(); ++c)
                      {
                        extractChannel<float>(C, _channels[c], c, _sigma_bp);
                      }
                    });
#else
#if defined(WITH_OPENMP)
#pragma omp parallel for
#endif
  for(size_t i = 0; i < NumChannels; ++i) {
    extractChannel<float>(C, _channels[i], i, _sigma_bp);
  }
#endif
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

