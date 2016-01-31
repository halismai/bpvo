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
  BitPlanesChannelMaker(const cv::Mat& C, float sigma, BitPlanesData& data)
      : _C(C), _sigma(sigma), _data(data) {}

  void operator()(const tbb::blocked_range<int>& range) const
  {
    for(int i = range.begin(); i != range.end(); ++i) {
      extractChannel<float>(_C, _data.cn[i], i);

      if(_sigma > 0.0)
        cv::GaussianBlur(_data.cn[i], _data.cn[i], cv::Size(5,5), _sigma, _sigma);
    }
  }

  const cv::Mat& _C;
  float _sigma;
  BitPlanesData& _data;
}; // BitPlanesChannelMaker

BitPlanesData computeBitPlanes(const cv::Mat& I, float s1, float s2)
{
  auto C = census(I, s1);

  BitPlanesData ret;

  BitPlanesChannelMaker bp(C, s2, ret);
  tbb::blocked_range<int> range(0, 8);

#if EXTRACT_CHANNELS_SERIAL
  tbb::serial::parallel_for(range, bp);
#else
  tbb::parallel_for(range, bp);
#endif

  return ret;
}

template <typename TSrc> static FORCE_INLINE
cv::Mat computeGradientAbsMagChannel(const cv::Mat& src)
{

  int rows = src.rows, cols = src.cols;
  cv::Mat dst(rows, cols, cv::DataType<TSrc>::type);

#if 0
  using namespace Eigen;
  typedef Matrix<TSrc, Dynamic, Dynamic, RowMajor> MatrixType;
  typedef Map<const MatrixType, Aligned> SrcMap;
  typedef Map<MatrixType, Aligned> DstMap;

  SrcMap I(src.ptr<const TSrc>(), rows, cols);
  DstMap G(dst.ptr<TSrc>(), rows, cols);

  // we are ignoring the borders here and proper normalization of the gradient
  // abs(Iy) + abs(Ix)
  // TODO check if Eigen is creating temporaries here
  G.block(1, 0, rows - 2, cols) =
      (I.block(0, 2, rows - 2, cols) -
       I.block(0, 0, rows - 2, cols)).array().abs() +
      (I.block(0, 2, rows, cols - 2) -
       I.block(0, 0, rows, cols - 2)).array().abs();
#endif

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
  assert( !cn.front().empty() );

  int rows = cn.front().rows, cols = cn.front().cols;
  gradientAbsMag = cv::Mat(rows, cols, CV_32FC1, cv::Scalar(0));

  for(const auto& c : cn) {
    accumulateGradientAbsMag<float>(c, gradientAbsMag);
  }
}

}; // bpvo

#if EXTRACT_CHANNELS_SERIAL
#undef TBB_PREVIEW_SERIAL_SUBSET
#endif

#undef EXTRACT_CHANNELS_SERIAL

