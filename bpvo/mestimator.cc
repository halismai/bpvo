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

#include "bpvo/mestimator.h"
#include "bpvo/approximate_median.h"
#include "bpvo/math_utils.h"
#include "bpvo/utils.h"

#include <algorithm>

#if defined(WITH_SIMD)
#include <immintrin.h>
#endif

namespace bpvo {

template <typename T = float>
struct HuberOp
{
  inline HuberOp(T k = T(1.345)) : _k(k) { }

  inline T operator()(T r) const noexcept
  {
    T x = std::fabs(r);
    return (x < _k) ? 1.0f : (_k / x);
  }

  T _k;
}; // HuberOp

template <typename T = float>
struct TukeyOp
{
  inline TukeyOp(T t = T(4.685)) : _t(t), _t_inv(1.0 / t) {}

  inline T operator()(T r) const noexcept
  {
    T x = std::fabs(r);
    return (x < 1e-6) ? 1.0 : (x > _t) ? 0.0f : math::sq(1.0f - math::sq(_t_inv * x));
  }

  T _t, _t_inv;
}; // TukeyOp


template <class RobustFunction, class... Args> static inline
void computeWeights(const ResidualsVector& residuals, float sigma,
                    WeightsVector& weights, Args ... args)
{
  RobustFunction robust_fn(std::forward<Args>(args)...);
  float sigma_inv = 1.0f / sigma;

  std::transform(residuals.begin(), residuals.end(), weights.begin(),
                 [=](float x) { return robust_fn(sigma_inv * x); });
}

template <class RobustFunction, class... Args> static inline
void computeWeights(const ResidualsVector& residuals, const ValidVector& valid,
                    float sigma, WeightsVector& weights, Args ... args)
{
  RobustFunction robust_fn(std::forward<Args>(args)...);

  float sigma_inv = 1.0f / sigma;
  for(size_t i = 0; i < valid.size(); ++i) {
    weights[i] = valid[i] ? robust_fn(sigma_inv * residuals[i]) : 0.0f;
  }
}

void MEstimator::
ComputeWeights(LossFunctionType loss_func, const ResidualsVector& residuals,
               float sigma, WeightsVector& weights)
{
  weights.resize(residuals.size());

  if(loss_func == LossFunctionType::kL2) {
    std::fill(weights.begin(), weights.end(), 1.0f);
    return;
  }

  switch(loss_func) {
    case LossFunctionType::kHuber: computeWeights<HuberOp<float>>(residuals, sigma, weights); break;
    case LossFunctionType::kTukey: computeWeights<TukeyOp<float>>(residuals, sigma, weights); break;
    default: THROW_ERROR("unkonwn RobustFunction");
  }
}

#if defined(WITH_SIMD)

#if defined(__AVX__)
static const __m256 SIGN_MASK_PS = _mm256_set1_ps(-0.f);

static FORCE_INLINE __m256 abs_ps(__m256 x) {
  return _mm256_andnot_ps(SIGN_MASK_PS, x);
}

static FORCE_INLINE __m256 set1_ps(float v)
{
  return _mm256_set1_ps(v);
}

static FORCE_INLINE __m256 load_ps(const float* ptr)
{
  return _mm256_load_ps(ptr);
}

static FORCE_INLINE __m256 loadu_ps(const float* ptr)
{
  return _mm256_loadu_ps(ptr);
}

static FORCE_INLINE __m256 sub_ps(__m256 a, __m256 b)
{
  return _mm256_sub_ps(a, b);
}

static FORCE_INLINE __m256 cmplt_ps(__m256 a, __m256 b)
{
  return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
}

static FORCE_INLINE __m256 mul_ps(__m256 a, __m256 b)
{
  return _mm256_mul_ps(a, b);
}

static FORCE_INLINE __m256 div_ps(__m256 a, __m256 b)
{
  return _mm256_div_ps(a, b);
}

static FORCE_INLINE __m256 and_ps(__m256 a, __m256 b)
{
  return _mm256_and_ps(a, b);
}

static FORCE_INLINE void store_ps(float* ptr, __m256 a)
{
  _mm256_store_ps(ptr, a);
}


static FORCE_INLINE void storeu_ps(float* ptr, __m256 a)
{
  _mm256_storeu_ps(ptr, a);
}

static FORCE_INLINE __m256 max_ps(__m256 a, __m256 b)
{
  return _mm256_max_ps(a, b);
}

static const int SIMD_VECTOR_UNIT_SIZE = 8;

#else
//
// SSE instructions
//
static const __m128 SIGN_MASK_PS = _mm_set1_ps(-0.0f);

static FORCE_INLINE __m128 abs_ps(__m128 x) {
  return _mm_andnot_ps(SIGN_MASK_PS, x);
}

static FORCE_INLINE __m128 load_ps(const float* ptr)
{
  return _mm_load_ps(ptr);
}

static FORCE_INLINE __m128 loadu_ps(const float* ptr)
{
  return _mm_loadu_ps(ptr);
}

static FORCE_INLINE __m128 mul_ps(__m128 a, __m128 b)
{
  return _mm_mul_ps(a, b);
}

static FORCE_INLINE void store_ps(float* ptr, __m128 a)
{
  _mm_store_ps(ptr, a);
}

static FORCE_INLINE void storeu_ps(float* ptr, __m128 a)
{
  _mm_storeu_ps(ptr, a);
}

static FORCE_INLINE __m128 max_ps(__m128 a, __m128 b)
{
  return _mm_max_ps(a, b);
}

static FORCE_INLINE __m128 set1_ps(float v)
{
  return _mm_set1_ps(v);
}

static FORCE_INLINE __m128 div_ps(__m128 a, __m128 b)
{
  return _mm_div_ps(a, b);
}

static FORCE_INLINE __m128 sub_ps(__m128 a, __m128 b)
{
  return _mm_sub_ps(a, b);
}

static FORCE_INLINE __m128 and_ps(__m128 a, __m128 b)
{
  return _mm_and_ps(a, b);
}

static FORCE_INLINE __m128 cmplt_ps(__m128 a, __m128 b)
{
  return _mm_cmplt_ps(a, b);
}


static const int SIMD_VECTOR_UNIT_SIZE = 4;

#endif // __AVX__

static inline
size_t huber_simd(const typename ResidualsVector::value_type* r_ptr,
                  const typename  ValidVector::value_type* /* v_ptr */,
                  typename WeightsVector::value_type* w_ptr,
                  size_t N, float sigma_inv, float huber_k)
{
  constexpr int S = 2 * SIMD_VECTOR_UNIT_SIZE;
  auto n = N & ~(S-1);
  const auto s_inv = set1_ps(sigma_inv), h_k = set1_ps(huber_k);

  size_t i = 0;
  bool is_aligned = false;

  is_aligned = IsAligned<DefaultAlignment>(r_ptr) && IsAligned<DefaultAlignment>(w_ptr);

  if(is_aligned) {
    for(i=0; i < n; i += S) {
      auto x0 = abs_ps( mul_ps(load_ps(r_ptr + i + 0*SIMD_VECTOR_UNIT_SIZE), s_inv) );
      auto x1 = abs_ps( mul_ps(load_ps(r_ptr + i + 1*SIMD_VECTOR_UNIT_SIZE), s_inv) );
      auto r0 = div_ps(h_k, max_ps(x0, h_k));
      auto r1 = div_ps(h_k, max_ps(x1, h_k));
      store_ps(w_ptr + i + 0*SIMD_VECTOR_UNIT_SIZE, r0);
      store_ps(w_ptr + i + 1*SIMD_VECTOR_UNIT_SIZE, r1);
    }
  } else {
    for(i=0; i < n; i += S) {
      auto x0 = abs_ps( mul_ps(loadu_ps(r_ptr + i + 0*SIMD_VECTOR_UNIT_SIZE), s_inv) );
      auto x1 = abs_ps( mul_ps(loadu_ps(r_ptr + i + 1*SIMD_VECTOR_UNIT_SIZE), s_inv) );
      auto r0 = div_ps(h_k, max_ps(x0, h_k));
      auto r1 = div_ps(h_k, max_ps(x1, h_k));
      storeu_ps(w_ptr + i + 0*SIMD_VECTOR_UNIT_SIZE, r0);
      storeu_ps(w_ptr + i + 1*SIMD_VECTOR_UNIT_SIZE, r1);
    }
  }

#if defined(__AVX__)
  _mm256_zeroupper();
#endif

  return i;
}

static inline
void computeWeightsHuberSimd(const ResidualsVector& residuals, const ValidVector& valid,
                             float sigma, WeightsVector& weights, float huber_k = 1.345f)
{
  assert( valid.size() == residuals.size() );
  weights.resize(residuals.size());

  float sigma_inv = 1.0f / sigma;
  size_t i = huber_simd(residuals.data(), valid.data(), weights.data(),
                        residuals.size(), sigma_inv, huber_k);

  if(i < residuals.size()) {
    HuberOp<float> func(huber_k);
    for( ; i < residuals.size(); ++i) {
      weights[i] = static_cast<float>(valid[i]) * func(sigma_inv * residuals[i]);
    }
  }
}

static inline
size_t tukey_simd(const typename ResidualsVector::value_type* r_ptr,
                  const typename ValidVector::value_type* /* v_ptr */,
                  typename ResidualsVector::value_type* w_ptr,
                  size_t N, float sigma_inv, float tukey_t)
{
  size_t i = 0;

  constexpr int S = 2 * SIMD_VECTOR_UNIT_SIZE;
  auto n = N & ~(S-1);
  const auto s_inv = set1_ps(sigma_inv), ones = set1_ps(1.0f),
        t = set1_ps(tukey_t), t_i = set1_ps(1.0 / tukey_t);

  bool is_aligned = false;
  is_aligned = IsAligned<DefaultAlignment>(r_ptr) && IsAligned<DefaultAlignment>(w_ptr);

  if(is_aligned) {
    for(i = 0; i < n; i += S) {
      {
        auto x = mul_ps(load_ps(r_ptr + i + 0*SIMD_VECTOR_UNIT_SIZE), s_inv);
        auto r = mul_ps(x, t_i);
        r = sub_ps(ones, mul_ps(r,r));
        r = mul_ps(r, r);
        auto m = cmplt_ps(abs_ps(x), t);
        store_ps(w_ptr + i + 0*SIMD_VECTOR_UNIT_SIZE, and_ps(m, r));
      }

      {
        auto x = mul_ps(load_ps(r_ptr + i + 1*SIMD_VECTOR_UNIT_SIZE), s_inv);
        auto r = mul_ps(x, t_i);
        r = sub_ps(ones, mul_ps(r,r));
        r = mul_ps(r, r);
        auto m = cmplt_ps(abs_ps(x), t);
        store_ps(w_ptr + i + 1*SIMD_VECTOR_UNIT_SIZE, and_ps(m, r));
      }
    }
  } else {
    for(i = 0; i < n; i += S) {
      {
        auto x = mul_ps(loadu_ps(r_ptr + i + 0*SIMD_VECTOR_UNIT_SIZE), s_inv);
        auto r = mul_ps(x, t_i);
        r = sub_ps(ones, mul_ps(r,r));
        r = mul_ps(r, r);
        auto m = cmplt_ps(abs_ps(x), t);
        storeu_ps(w_ptr + i + 0*SIMD_VECTOR_UNIT_SIZE, and_ps(m, r));
      }

      {
        auto x = mul_ps(loadu_ps(r_ptr + i + 1*SIMD_VECTOR_UNIT_SIZE), s_inv);
        auto r = mul_ps(x, t_i);
        r = sub_ps(ones, mul_ps(r,r));
        r = mul_ps(r, r);
        auto m = cmplt_ps(abs_ps(x), t);
        storeu_ps(w_ptr + i + 1*SIMD_VECTOR_UNIT_SIZE, and_ps(m, r));
      }
    }
  }

#if defined(__AVX__)
  _mm256_zeroupper();
#endif

  return i;
}

static inline
void computeWeightsTukeySimd(const ResidualsVector& residuals,
                             const ValidVector& valid,
                             float sigma, WeightsVector& weights)
{
  assert( valid.size() == residuals.size() );
  weights.resize(valid.size());

  float sigma_inv = 1.0f / sigma;
  size_t i = tukey_simd(residuals.data(), valid.data(), weights.data(),
                        residuals.size(), sigma_inv, 4.685f);

  if(i < residuals.size()) {
    TukeyOp<float> func(4.685f);
    for( ; i < residuals.size(); ++i)
      weights[i] = static_cast<float>(valid[i]) * func(sigma_inv * residuals[i]);
  }
}


#endif // WITH_SIMD

void MEstimator::
ComputeWeights(LossFunctionType loss_func, const ResidualsVector& residuals,
               const ValidVector& valid, float sigma, WeightsVector& weights)
{
  assert( residuals.size() == valid.size() );
  weights.resize(valid.size());

  if(loss_func == LossFunctionType::kL2) {
    std::fill(weights.begin(), weights.end(), 1.0f);
    return;
  }

#if defined(WITH_SIMD)
  switch(loss_func) {
    case LossFunctionType::kHuber: computeWeightsHuberSimd(residuals, valid, sigma, weights); break;
    case LossFunctionType::kTukey: computeWeightsTukeySimd(residuals, valid, sigma, weights); break;
    default: THROW_ERROR("unknown RobustFunction");
  }
#else
  switch(loss_func) {
    case LossFunctionType::kHuber: computeWeights<HuberOp<float>>(residuals, valid, sigma, weights); break;
    case LossFunctionType::kTukey: computeWeights<TukeyOp<float>>(residuals, valid, sigma, weights); break;
    default: THROW_ERROR("unkonwn RobustFunction");
  }
#endif
}

#if DO_APPROX_MEDIAN
AutoScaleEstimator::AutoScaleEstimator(float t)
  : _scale(1.0), _delta_scale(1e10), _tol(t), _hist(0.0f, 255.0f, 0.05f) {}
#else
AutoScaleEstimator::AutoScaleEstimator(float t)
  : _scale(1.0), _delta_scale(1e10), _tol(t) {}
#endif

void AutoScaleEstimator::reset()
{
  _delta_scale = 1e10;
  _scale = 1.0;

#if DO_APPROX_MEDIAN
  _hist.clear();
#endif

}

float AutoScaleEstimator::getScale() const { return _scale; }

static inline
float ScaleEstimator(const ResidualsVector& residuals,
                     const ValidVector& valid_flags,
                     Histogram<float>& hist)
{
  hist.clear();

  for(size_t i = 0; i < residuals.size(); ++i)
    if(valid_flags[i] != 0)
      hist.add(std::fabs(residuals[i]));

  return (1.4826f * (1.0f + 5.0f / (hist.numSamples()-6) )) * hist.median();
}

static inline
float ScaleEstimator(const ResidualsVector& residuals,
                     const ValidVector& valid_flags,
                     WeightsVector& buffer)
{
  buffer.resize(0);
  buffer.reserve(residuals.size());

  for(size_t i = 0; i < residuals.size(); ++i)
    if(valid_flags[i] != 0)
      buffer.push_back( std::fabs(residuals[i]) );

  return (1.4826f * (1.0f + 5.0f / (buffer.size()-6) )) * median(buffer);
}

float AutoScaleEstimator::estimateScale(const ResidualsVector& residuals,
                                        const ValidVector& valid)
{
  assert( residuals.size() == valid.size() );

  if(_delta_scale > _tol)
  {
#if DO_APPROX_MEDIAN
    auto scale = ScaleEstimator(residuals, valid, _hist);
#else
    auto scale = ScaleEstimator(residuals, valid, _buffer);
#endif

    if(scale < 1e-6)
      scale = 1.0; // for the case of zero error

    _delta_scale = std::fabs(scale - _scale);
    _scale = scale;
  } else {
    dprintf("scale is stable %f\n", _delta_scale);
  }

  return _scale;
}

}; // bpvo


