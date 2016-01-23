#include "bpvo/bitplanes.h"
#include "bpvo/debug.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstring>
#include <iostream>
#include <type_traits>

#include <emmintrin.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace bpvo {

/**
 * Holds a vector of 16 bytes (128 bits)
 */
struct v128
{
  __m128i _xmm; //< the vector

  FORCE_INLINE v128() {}

  /**
   * loads the data from vector (unaligned load)
   */
  FORCE_INLINE v128(const uint8_t* p)
      : _xmm(_mm_loadu_si128((const __m128i*)p)) {}

  /**
   * assign from __m128i
   */
  FORCE_INLINE v128(__m128i x) : _xmm(x) {}

  FORCE_INLINE const v128& load(const uint8_t* p)
  {
    _xmm = _mm_load_si128(( __m128i*) p);
    return *this;
  }

  FORCE_INLINE const v128& loadu(const uint8_t* p)
  {
    _xmm = _mm_loadu_si128((__m128i*) p);
    return *this;
  }

  /**
   * set to constant
   */
  FORCE_INLINE v128(int n) : _xmm(_mm_set1_epi8(n)) {}

  FORCE_INLINE operator __m128i() const { return _xmm; }
  //FORCE_INLINE operator const __m128i&() const { return _xmm; }

  FORCE_INLINE v128 Zero()    { return _mm_setzero_si128(); }
  FORCE_INLINE v128 InvZero() { return v128(0xff); }
  FORCE_INLINE v128 One()     { return v128(0x01); }

  FORCE_INLINE void store(const void* p) const
  {
    _mm_store_si128((__m128i*) p, _xmm);
  }

  friend std::ostream& operator<<(std::ostream&, const v128&);
}; // v128

/**
 */
FORCE_INLINE v128 max(v128 a, v128 b)
{
  return _mm_max_epu8(a, b);
}

FORCE_INLINE v128 min(v128 a, v128 b)
{
  return _mm_min_epu8(a, b);
}

FORCE_INLINE v128 operator==(v128 a, v128 b)
{
  return _mm_cmpeq_epi8(a, b);
}

FORCE_INLINE v128 operator>=(v128 a, v128 b)
{
  return (a == max(a, b));
}

FORCE_INLINE v128 operator>(v128 a, v128 b)
{
  return _mm_andnot_si128( min(a, b) == a, _mm_set1_epi8(0xff) );
}

FORCE_INLINE v128 operator<(v128 a, v128 b)
{
  return _mm_andnot_si128( max(a, b) == a, _mm_set1_epi8(0xff) );
}

FORCE_INLINE v128 operator<=(v128 a, v128 b)
{
  return (a == min(a, b));
}

FORCE_INLINE v128 operator&(v128 a, v128 b)
{
  return _mm_and_si128(a, b);
}

FORCE_INLINE v128 operator|(v128 a, v128 b)
{
  return _mm_or_si128(a, b);
}

FORCE_INLINE v128 operator^(v128 a, v128 b)
{
  return _mm_xor_si128(a, b);
}

template <int imm> FORCE_INLINE v128 SHIFT_RIGHT(v128 a)
{
  return _mm_srli_epi32(a, imm);
}

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

static cv::Mat census(const cv::Mat& src, float s)
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

template <typename TSrc, typename TDst>
inline void computeImageGradientPacked(const cv::Mat& src, cv::Mat& dst)
{
  // for now, only floating point TDst
  static_assert(std::is_floating_point<TDst>::value, "dest must be floating point");

  assert( cv::DataType<TSrc>::type == src.type() );
  assert( !src.empty() );

  dst.create(src.size(), CV_MAKETYPE(cv::DataType<TDst>::depth, 2));

  // we'll skip the border
  for(int y = 1; y < src.rows - 2; ++y) {
    auto srow0 = src.ptr<const TSrc>(y-1);
    auto srow1 = src.ptr<const TSrc>(y+1);
    auto srow = src.ptr<const TSrc>(y);

    auto drow = dst.ptr<TDst>(y);

#pragma omp simd
    for(int x = 1; x < src.cols - 2; ++x) {
      drow[2*x + 0] = TDst(0.5) * ((TDst) srow[x+1] - (TDst) srow[x]);
      drow[2*x + 1] = TDst(0.5) * ((TDst) srow1[x] - (TDst) srow0[x]);
    }
  }
}

void BitPlanesData::computeGradients(int i)
{
  computeImageGradientPacked<float, float>(I[i], G[i]);
}

template <typename DstType>
void extractChannel(const cv::Mat& C, cv::Mat& BP, int b)
{
  BP.create(C.size(), cv::DataType<DstType>::type);
  auto src = C.ptr<const uint8_t>();
  auto dst = BP.ptr<DstType>();

#pragma omp simd
  for(int i = 0; i < C.rows*C.cols; ++i) {
    dst[i] = static_cast<DstType>( (src[i] & (1 << b)) >> b );
  }

}

struct BitPlanesChannelMaker
{
  BitPlanesChannelMaker(const cv::Mat& C, float sigma, BitPlanesData& data)
      : _C(C), _sigma(sigma), _data(data) {}

  void operator()(const tbb::blocked_range<int>& range) const
  {
    for(int i = range.begin(); i != range.end(); ++i)
    {
      extractChannel<float>(_C, _data.I[i], i);

      if(_sigma > 0.0) {
        cv::GaussianBlur(_data.I[i], _data.I[i], cv::Size(5,5), _sigma, _sigma);
      }

      _data.computeGradients(i);
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
  tbb::parallel_for(tbb::blocked_range<int>(0,8), bp);

  return ret;
}

}; // bpvo
