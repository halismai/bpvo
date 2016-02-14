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

#include "bpvo/linear_system_builder_reduction.h"

#if defined(WITH_TBB)
#include <tbb/parallel_reduce.h>
#endif

#if defined(WITH_SIMD)
#include <pmmintrin.h>
#endif

namespace bpvo {

LinearSystemBuilderReduction::
LinearSystemBuilderReduction(const JacobianVector& J, const ResidualsVector& R,
                             const ResidualsVector& W, const ValidVector& V)
  : _J(J), _R(R), _W(W), _valid(V) { setZero(); }

LinearSystemBuilderReduction::~LinearSystemBuilderReduction() {}

#if defined(WITH_TBB)
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

  Gradient g_tmp(Gradient::Zero());
  float res_sq_norm = 0.0f;
  for(int i = range.begin(); i != range.end(); ++i)
      rankUpdatePoint(i, h_data, g_tmp, res_sq_norm);

#if defined(WITH_SIMD)
  _H.noalias() += LinearSystemBuilderReduction::toEigen(h_data);
#else
  _H.noalias() += h_tmp;
#endif
  _G.noalias() += g_tmp;
  _res_sq_norm += res_sq_norm;
}
#endif

void LinearSystemBuilderReduction::setZero()
{
  _H.setZero();
  _G.setZero();
  _res_sq_norm = 0.0f;
}

void LinearSystemBuilderReduction::rankUpdatePoint(int i, float* data, Gradient& G,
                                                   float& res_norm)

{
  float w = _W[i] * static_cast<float>(_valid[i]);

#if defined(WITH_SIMD)
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
#else
  typedef Eigen::Map<Hessian, Eigen::Aligned> HessianMap;
  HessianMap(data).noalias() += _W[i] * _J[i].transpose() * _J[i];
#endif

  G.noalias() += w * _R[i] * _J[i].transpose();
  res_norm += w * _R[i] * _R[i];
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

#if defined(WITH_BITPLANES) && defined(WITH_TBB)
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
    alignas(16) float data[24];
    std::fill_n(data, 24, 0.0f);
    h_data = data;
#else
    h_data = H->data();
#endif

    for(size_t i = 0; i < R.size(); ++i) {
      reduction.rankUpdatePoint(i, h_data, *G, ret);
    }

#if defined(WITH_SIMD)
    *H = LinearSystemBuilderReduction::toEigen(h_data);
#endif

    return ret;
#endif
  } else {
    // for sum of squares, it is faster to not parallarize
    float ret = 0.0f;

    const float* r_ptr = R.data();
    const float* w_ptr = W.data();
    const uint8_t* v_ptr = V.data();

#if defined(WITH_OPENMP)
#pragma omp simd
#endif
    for(size_t i = 0; i < R.size(); ++i) {
        ret += static_cast<float>(v_ptr[i]) * w_ptr[i] * r_ptr[i] * r_ptr[i];
    }

    return ret;
  }
}

}; // bpvo
