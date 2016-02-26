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


#ifndef BPVO_LINEAR_SYSTEM_BUILDER_REDUCTION_H
#define BPVO_LINEAR_SYSTEM_BUILDER_REDUCTION_H

#include <bpvo/linear_system_builder.h>

#if defined(WITH_TBB)
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

namespace bpvo {

class LinearSystemBuilderReduction
{
 public:
  typedef typename LinearSystemBuilder::JacobianVector  JacobianVector;
  typedef typename LinearSystemBuilder::Hessian         Hessian;
  typedef typename LinearSystemBuilder::Gradient        Gradient;

  typedef typename ResidualsVector::value_type ResidualsType;
  typedef typename WeightsVector::value_type   WeightsType;
  typedef typename ValidVector::value_type     ValidType;

 public:
  inline
  LinearSystemBuilderReduction(const JacobianVector& J, const ResidualsType* R,
                               const WeightsType* W, const ValidVector* V)
  : _J(J), _R(R), _W(W), _V(V)
  , _H(Hessian::Zero()), _G(Gradient::Zero()), _res_sq_norm(0.0) {}

  inline
  LinearSystemBuilderReduction(LinearSystemBuilderReduction& o, tbb::split)
      : _J(o._J), _R(o._R), _W(o._w), _V(o._v)
      , _H(Hessian::Zero()), _G(Gradient::Zero()), _res_sq_norm(0.0f) {}

  inline
  void join(const LinearSystemBuilderReduction& o)
  {
    _H.noalias() += o._H;
    _G.noalias() += o._G;
    _res_sq_norm += o._res_sq_norm;
  }

  inline const Hessian& hessian() const { return _H; }
  inline const Gradient& gradient() const { return _G; }
  inline const float& residualsSquaredNorm() const { return _res_sq_norm; }

  inline
  void operator()(const tbb::blocked_range<int>& range)
  {
    alignas(16) float data[24];
    memset(data, 0, sizeof(data));

    float res_sq_norm = _res_sq_norm;

    for(int i = range.begin(); i != range.end(); ++i)
    {
      float w = _W[i] * _valid[i],
            wR = _W[i] * _R[i];

      res_sq_norm += wR * _R[i];

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

      G.noalias() += wR * _J[i].transpose();
    }

    _res_sq_norm = res_sq_norm;
  }

  static inline float Run(const JacobianVector& J, const ResidualsVector& R,
                         const WeightsVector& W, const ValidVector& V,
                         Hessian& H, Gradient& G)
  {
    assert( R.size() == W.size() && R.size() == V.size() );
    assert( (J.size()-1) == R.size() );

    LinearSystemBuilderReduction func(J, R.data(), W.data(), V.data());
    tbb::parallel_reduce(tbb::blocked_range<int>(0, (int) R.size()), func);

    H = func.hessian();
    G = func.gradient();
    return func.residualsSquaredNorm();
  }

 protected:
  const JacobianVector& _J;
  const ResidualsType* _R;
  const WeightsType* _W;
  const ValidType* _V;

  Hessian _H;
  Gradient _G;
  float _res_sq_norm = 0.0f;

}; // LinearSystemBuilderReduction

}; // bpvo
#else

#endif // WITH_TBB



#endif // BPVO_LINEAR_SYSTEM_BUILDER_REDUCTION_H
