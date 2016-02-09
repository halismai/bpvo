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

#ifndef BPVO_LINEAR_SYSTEM_REDUCTION_H
#define BPVO_LINEAR_SYSTEM_REDUCTION_H

#include <bpvo/linear_system_builder.h>

#if defined(WITH_TBB)
#include <tbb/blocked_range.h>
#endif // WITH_TBB

#include <vector>

namespace bpvo {

class LinearSystemBuilderReduction
{
 public:
  typedef typename LinearSystemBuilder::JacobianVector  JacobianVector;
  typedef typename LinearSystemBuilder::ResidualsVector ResidualsVector;
  typedef typename LinearSystemBuilder::Hessian         Hessian;
  typedef typename LinearSystemBuilder::Gradient        Gradient;
  typedef std::vector<uint8_t>                          ValidVector;

 public:
  LinearSystemBuilderReduction(const JacobianVector& J, const ResidualsVector& R,
                               const ResidualsVector& W, const ValidVector& V);
  ~LinearSystemBuilderReduction();

#if defined(WITH_TBB)
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

  void rankUpdatePoint(int i, float* H_data, Gradient& G, float& res_norm);

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


}; // bpvo

#endif // BPVO_LINEAR_SYSTEM_REDUCTION_H
