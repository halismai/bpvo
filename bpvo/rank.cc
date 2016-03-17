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


#include "bpvo/rank.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <array>
#include <vector>
#include <bitset>

namespace bpvo {

cv::Mat rankTransform(const cv::Mat& src, float sigma)
{
  cv::Mat I = src;
  if(sigma > 0)
    cv::GaussianBlur(I, I, cv::Size(3,3), sigma, sigma);

  typedef uint8_t DstType;
  cv::Mat ret(src.size(), cv::DataType<DstType>::type);
  memset(ret.data, 0, ret.cols * sizeof(DstType));

  for(int r = 1; r < ret.rows - 1; ++r)
  {
    auto srow = src.ptr<uint8_t>(r),
         srow0 = src.ptr<uint8_t>(r-1),
         srow1 = src.ptr<uint8_t>(r+1);
    auto drow = ret.ptr<DstType>(r);
    drow[0] = DstType(0);

#if defined(WITH_OPENMP)
#pragma omp simd
#endif
    for(int c = 1; c < ret.cols - 1; ++c)
    {
      const auto v = srow[c];
      drow[c] =
          (srow0[-1] > v) + (srow0[0] > v) + (srow0[1] > v) +
          (srow [-1] > v)                  + (srow [1] > v) +
          (srow1[-1] > v) + (srow[1] > v)  + (srow1[1] > v);
    }
    drow[ret.cols-1] = DstType(0);
  }

  memset(ret.ptr<DstType>(ret.rows-1), 0, ret.cols*sizeof(DstType));

  return ret;
}

template <int N> static inline
std::array<uint8_t, N> computeFullRank(const std::array<uint8_t,N>& p)
{
  std::array<uint8_t, N> ret;
  for(int i = 0; i < N; ++i)
  {
    uint8_t r = 0;
    for(int j = 0; j < N; ++j)
    {
      if(i != j)
      {
        r += p[j] < p[i];
      }
    }
    ret[i] = r;
  }

  return ret;
}

void completeRankPlanes(const cv::Mat& src, std::array<cv::Mat,9>& dst,
                        float sigma, float sigma_rank)
{
  cv::Mat I = src;
  if(sigma > 0)
    cv::GaussianBlur(I, I, cv::Size(3,3), sigma, sigma);

  std::vector<std::array<uint8_t,9>> rank(src.cols*src.cols);
  for(int r = 1; r < src.rows - 1; ++r)
  {
    auto srow = src.ptr<uint8_t>(r),
         srow0 = src.ptr<uint8_t>(r-1),
         srow1 = src.ptr<uint8_t>(r+1);
    for(int c = 1; c < src.cols - 1; ++c)
    {
      std::array<uint8_t,9> P{
        {srow0[-1], srow0[0], srow0[1],
         srow [-1], srow [0], srow [1],
         srow1[-1], srow1[0], srow1[1]}};

      rank[r*src.cols + c] = computeFullRank<9>(P);
    }
  }

  for(int i = 0; i < 9; ++i)
  {
    typedef float DstType;
    dst[i].create(src.size(), cv::DataType<DstType>::type);
    memset(dst[i].data, 0, sizeof(DstType) * src.cols);
    for(int r = 1; r < src.rows-1; ++r) {
      dst[i].at<DstType>(r,0) = DstType(0);
      for(int c = 1; c < src.cols-1; ++c) {
        dst[i].at<DstType>(r,c) = (DstType) rank[r*src.cols+c][i];
      }
      dst[i].at<DstType>(r,src.cols-1) = DstType(0);
    }
    memset(dst[i].ptr<DstType>(src.rows-1), 0, sizeof(DstType) * src.cols);

    if(sigma_rank > 0.0)
      cv::GaussianBlur(dst[i], dst[i], cv::Size(3,3), sigma_rank, sigma_rank);
  }
}

} // bpvo
