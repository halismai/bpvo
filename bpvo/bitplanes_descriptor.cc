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

#include "bpvo/bitplanes_descriptor.h"
#include "bpvo/census.h"
#include "bpvo/parallel.h"
#include "bpvo/imgproc.h"
#include "bpvo/utils.h"

#include <opencv2/imgproc/imgproc.hpp>

namespace bpvo {

BitPlanesDescriptor::BitPlanesDescriptor(float s0, float s1)
    : _rows(0), _cols(0), _sigma_ct(s0), _sigma_bp(s1) {}

BitPlanesDescriptor::~BitPlanesDescriptor() {}

template <typename TDst> static inline
void ExtractChannel(const cv::Mat& src, cv::Mat& dst, int bit, float sigma)
{
  dst.create(src.size(),  cv::DataType<TDst>::type);

  constexpr TDst Scale = TDst(1.0); // to keeep thresholds the same
  constexpr TDst Bias  = TDst(0.0);

  auto src_ptr = src.ptr<const uint8_t>();
  auto dst_ptr = dst.ptr<TDst>();
  auto n = src.rows * src.cols;

#if defined(WITH_OPENMP)
#pragma omp simd
#endif
  for(int i = 0; i < n; ++i)
    dst_ptr[i] = Scale * ((src_ptr[i] & (1 << bit)) >> bit) - Bias;

  if(sigma > 0.0f)
    cv::GaussianBlur(dst, dst, cv::Size(5,5), sigma, sigma);
}

template <typename TDst>
class BitPlanesComputeBody : public ParallelForBody
{
 public:
  BitPlanesComputeBody(const cv::Mat& I, float sigma_ct, float sigma_bp,
                       std::array<cv::Mat,8>& bp)
      : ParallelForBody()
      , _C( census(I, sigma_ct) )
      , _sigma_bp(sigma_bp)
      , _channels(bp) {}

  virtual ~BitPlanesComputeBody() {}

  void operator()(const Range& range) const
  {
    for(int b = range.begin(); b != range.end(); ++b)
      ExtractChannel<TDst>(_C, _channels[b], b, _sigma_bp);
  }

 protected:
  cv::Mat _C;
  float _sigma_bp;
  std::array<cv::Mat,8>& _channels;
}; // BitPlanesComputeBody

void BitPlanesDescriptor::compute(const cv::Mat& I_)
{
  _rows = I_.rows;
  _cols = I_.cols;

  BitPlanesComputeBody<float> func(I_, _sigma_ct, _sigma_bp, _channels);
  parallel_for(Range(0, 8), func);
}

} // bpvo

