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

#ifndef LINEAR_SYSTEM_BUILDER_H
#define LINEAR_SYSTEM_BUILDER_H

#include <bpvo/types.h>
#include <bpvo/warps.h>

namespace bpvo {

class LinearSystemBuilder
{
 public:
  typedef typename detail::warp_traits<RigidBodyWarp>::JacobianVector JacobianVector;
  typedef typename detail::warp_traits<RigidBodyWarp>::Jacobian       Jacobian;

  typedef Eigen::Matrix<float, 6, 1> Gradient;
  typedef Eigen::Matrix<float, 6, 6> Hessian;

 public:

  /**
   * \param J the jacobians per pixel
   * \param R the residuals
   * \param weighted M-estimator weights
   * \param valid incidates which elements are valid. Invalid elements are the
   * ones that project outside the image and we do not need to add them to the
   * optimization. NOTE: their weight will be 0
   *
   * \param H = J'*W*J
   * \param G = J'*W*R
   * \return the norm of the weighted residuals
   */
  static float Run(const JacobianVector& J, const ResidualsVector& R,
                   const ResidualsVector& weights, const ValidVector& valid,
                   Hessian* = nullptr, Gradient* = nullptr);

}; // LinearSystemBuilder

}; // bpvo

#endif // LINEAR_SYSTEM_BUILDER_H
