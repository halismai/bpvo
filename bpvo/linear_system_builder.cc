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

#include "bpvo/linear_system_builder.h"
#include "bpvo/parallel.h"

#define LINEAR_SYSTEM_PARALLEL 1
#define DO_PARALLEL defined(WITH_TBB) && LINEAR_SYSTEM_PARALLEL

#if DO_PARALLEL
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#endif

#if defined(WITH_SIMD)
#include <immintrin.h>
#endif


namespace bpvo {

class LinearSystemBuilderReduction
{
 public:
  typedef typename LinearSystemBuilder::JacobianVector  JacobianVector;
  typedef typename LinearSystemBuilder::Hessian         Hessian;
  typedef typename LinearSystemBuilder::Gradient        Gradient;

 public:
  LinearSystemBuilderReduction(const JacobianVector& J, const ResidualsVector& R,
                               const ResidualsVector& W, const ValidVector& V);
  ~LinearSystemBuilderReduction();

#if DO_PARALLEL
  LinearSystemBuilderReduction(LinearSystemBuilderReduction&, tbb::split);
  void join(const LinearSystemBuilderReduction&);
  void operator()(const tbb::blocked_range<int>& range);
#endif

  inline const Hessian&   hessian()   const { return _H; }
  inline const Gradient&  gradient()  const { return _G; }
  inline const float& residualsSquaredNorm() const { return _res_sq_norm; }

  static float Run(const JacobianVector& J, const ResidualsVector& R,
                   const ResidualsVector& W, const ValidVector& V,
                   Hessian* H = nullptr, Gradient* G = nullptr);

  FORCE_INLINE void rankUpdatePoint(int i, float* H_data, float* G_data, float& res_norm);

 protected:
  const JacobianVector& _J;
  const ResidualsVector& _R;
  const ResidualsVector& _W;
  const ValidVector& _valid;

  Hessian _H = Hessian::Zero();
  Gradient _G = Gradient::Zero();
  float _res_sq_norm = 0.0f;

  void setZero();

  static Hessian toEigen(const float*);

}; // LinearSystemBuilderReduction

LinearSystemBuilderReduction::
LinearSystemBuilderReduction(const JacobianVector& J, const ResidualsVector& R,
                             const ResidualsVector& W, const ValidVector& V)
  : _J(J), _R(R), _W(W), _valid(V) { setZero(); }

LinearSystemBuilderReduction::~LinearSystemBuilderReduction() {}

#if DO_PARALLEL
LinearSystemBuilderReduction::
LinearSystemBuilderReduction(LinearSystemBuilderReduction& o, tbb::split)
: _J(o._J), _R(o._R), _W(o._W), _valid(o._valid) { setZero(); }

void LinearSystemBuilderReduction::join(const LinearSystemBuilderReduction& o)
{
  _H.noalias() += o._H;
  _G.noalias() += o._G;
  _res_sq_norm += o._res_sq_norm;
}

void LinearSystemBuilderReduction::operator()(const tbb::blocked_range<int>& range)
{
  float* h_data = nullptr;

#if defined(WITH_SIMD)
  alignas(16) float _data[24];
  std::fill_n(_data, 24, 0.0f);
  h_data = _data;
#else
  Hessian h_tmp(Hessian::Zero());
  h_data = h_tmp.data();
#endif

  alignas(16) float G_data[8];
  std::fill_n(G_data, 8, 0.0f);

  float res_sq_norm = 0.0f;
  for(int i = range.begin(); i != range.end(); ++i)
      rankUpdatePoint(i, h_data, G_data, res_sq_norm);

#if defined(WITH_SIMD)
  _H.noalias() += LinearSystemBuilderReduction::toEigen(h_data);
#else
  _H.noalias() += h_tmp;
#endif
  _G.noalias() += Eigen::Map<Gradient, Eigen::Aligned>(G_data);
  _res_sq_norm += res_sq_norm;
}
#endif

void LinearSystemBuilderReduction::setZero()
{
  _H.setZero();
  _G.setZero();
  _res_sq_norm = 0.0f;
}

FORCE_INLINE void LinearSystemBuilderReduction::
rankUpdatePoint(int i, float* data, float* G, float& res_norm)
{
  float w = _W[i] * static_cast<float>( _valid[i] );
  float wR = w *_R[i];

#if defined(WITH_SIMD)
  // this reduction is based on DVO SLAM by Christian Kerl.
  /**
   *  This file is part of dvo.
   *
   *  Copyright 2012 Christian Kerl <christian.kerl@in.tum.de> (Technical University of Munich)
   *  For more information see <http://vision.in.tum.de/data/software/dvo>.
   *
   *  dvo is free software: you can redistribute it and/or modify
   *  it under the terms of the GNU General Public License as published by
   *  the Free Software Foundation, either version 3 of the License, or
   *  (at your option) any later version.
   *
   *  dvo is distributed in the hope that it will be useful,
   *  but WITHOUT ANY WARRANTY; without even the implied warranty of
   *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   *  GNU General Public License for more details.
   *
   *  You should have received a copy of the GNU General Public License
   *  along with dvo.  If not, see <http://www.gnu.org/licenses/>.
   */

  __m128 wwww = _mm_set1_ps(w);
  __m128 v1234 = _mm_loadu_ps(_J[i].data());
  __m128 v56xx = _mm_loadu_ps(_J[i].data() + 4);

  __m128 v1212 = _mm_movelh_ps(v1234, v1234);
  __m128 v3434 = _mm_movehl_ps(v1234, v1234);
  __m128 v5656 = _mm_movelh_ps(v56xx, v56xx);

  __m128 v1122 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v1212, v1212));

  _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), _mm_mul_ps(v1122, v1212)));
  _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), _mm_mul_ps(v1122, v3434)));
  _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), _mm_mul_ps(v1122, v5656)));

  __m128 v3344 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v3434, v3434));

  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), _mm_mul_ps(v3344, v3434)));
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), _mm_mul_ps(v3344, v5656)));

  __m128 v5566 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v5656, v5656));
  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), _mm_mul_ps(v5566, v5656)));

  __m128 g1 = _mm_load_ps(G);
  __m128 g2 = _mm_load_ps(G + 4);
  __m128 wr = _mm_mul_ps(wwww, _mm_set1_ps(_R[i]));

  _mm_store_ps(G, _mm_add_ps(g1, _mm_mul_ps(wr, v1234)));
  _mm_store_ps(G+4, _mm_add_ps(g2, _mm_mul_ps(wr, v56xx)));

#else
  typedef Eigen::Map<Hessian, Eigen::Aligned> HessianMap;
  typedef Eigen::Map<Gradient, Eigen::Aligned> GradientMap;
  HessianMap(data).noalias() += _W[i] * _J[i].transpose() * _J[i];
  GradientMap(G).noalias() += wR[i] * _J[i].transpose();
#endif

  res_norm += wR * _R[i];
}

auto LinearSystemBuilderReduction::toEigen(const float* data) -> Hessian
{
  Hessian ret;
  for(int i = 0, ii=0; i < 6; i += 2) {
    for(int j = i; j < 6; j += 2) {
      ret(i  , j  ) = data[ii++];
      ret(i  , j+1) = data[ii++];
      ret(i+1, j  ) = data[ii++];
      ret(i+1, j+1) = data[ii++];
    }
  }

  ret.selfadjointView<Eigen::Upper>().evalTo(ret);
  return ret;
}

float LinearSystemBuilderReduction::
Run(const JacobianVector& J, const ResidualsVector& R, const ResidualsVector& W,
    const ValidVector& V, Hessian* H, Gradient* G)
{
  assert( R.size() == W.size() && R.size() == V.size() );
  assert( J.size()-1 && R.size() );

  if(H && G) {
    LinearSystemBuilderReduction reduction(J, R, W, V);

#if DO_PARALLEL
    tbb::parallel_reduce(tbb::blocked_range<int>(0, (int) R.size()), reduction);
    *H = reduction.hessian();
    *G = reduction.gradient();
    return reduction.residualsSquaredNorm();
#else
    H->setZero();
    G->setZero();
    float ret = 0.0f;

    float* h_data = nullptr;

#if defined(WITH_SIMD)
    ALIGNED(16) float data[24];
    std::fill_n(data, 24, 0.0f);
    h_data = data;
#else
    h_data = H->data();
#endif

    ALIGNED(16) float G_data[8];
    std::fill_n(G_data, 8, 0.0f);

    for(size_t i = 0; i < R.size(); ++i) {
      reduction.rankUpdatePoint(i, h_data, G_data, ret);
    }

#if defined(WITH_SIMD)
    *H = LinearSystemBuilderReduction::toEigen(h_data);
#endif

    memcpy(G->data(), G_data, 6 * sizeof(float));

    return ret;
#endif
  } else {
    // for sum of squares, it is faster to not parallarize
    auto ret = 0.0f;
    const auto* r_ptr = R.data();
    const auto* w_ptr = W.data();
    const auto* v_ptr = V.data();

#if defined(WITH_OPENMP)
#pragma omp simd
#endif
    for(size_t i = 0; i < R.size(); ++i) {
        ret += static_cast<float>(v_ptr[i]) * w_ptr[i] * r_ptr[i] * r_ptr[i];
    }

    return ret;
  }
}


static inline void getValidResiduals(const std::vector<uint8_t>& valid,
                                     const std::vector<float>& residuals,
                                     std::vector<float>& valid_residuals)
{
#define USE_ALL_DATA 0
  // residuals size is 8 times the size of valid
  // valid is stored per channel, so we loop 8 times!
  valid_residuals.resize(0);

#if USE_ALL_DATA
  valid_residuals.reserve(residuals.size());
  auto* ptr = residuals.data();

  for(int b = 0; b < 8; ++b) {
    for(size_t i=0; i < valid.size(); ++i, ++ptr)
      if(valid[i])
        valid_residuals.push_back(*ptr);
  }
#else
  // most of the other residuals will look the same. We'll cut the cost by
  // inspecting a single channel only
  valid_residuals.reserve(valid.size());
  for(size_t i = 0; i < valid.size(); ++i)
    if(valid[i])
      valid_residuals.push_back(residuals[i]);

#endif
#undef USE_ALL_DATA
}

/**
 * makes the valid vector flags the same size as the number of residuals
 * NC is the number of channels of the descriptor
 */
static inline
ValidVector makeValidFlags(const ValidVector& v, int NC = 8)
{
  ValidVector ret(v.size()*NC);
  auto* p = ret.data();

  for(int b = 0; b < NC; ++b)
    for(size_t i = 0; i < v.size(); ++i)
      *p++ = v[i];

  return ret;
}

float LinearSystemBuilder::Run(const JacobianVector& J, const ResidualsVector& residuals,
                               const ResidualsVector& weights, const ValidVector& valid,
                               Hessian* A, Gradient* b)
{
  auto nc = residuals.size() / valid.size();
  assert( (J.size()-1) == residuals.size() && valid.size() == residuals.size()/nc );

  auto valid2 = residuals.size() != valid.size() ? makeValidFlags(valid, nc) : valid;
  assert( valid2.size() == residuals.size() );

#if defined(__AVX__)
  _mm256_zeroupper();
#endif

  float res_sq_norm = LinearSystemBuilderReduction::Run(J, residuals, weights, valid, A, b);
  return std::sqrt(res_sq_norm);
}

}; // bpvo

#undef DO_PARALLEL
#undef LINEAR_SYSTEM_PARALLEL

