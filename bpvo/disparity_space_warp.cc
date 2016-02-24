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

#include "bpvo/disparity_space_warp.h"

namespace bpvo {

DisparitySpaceWarp::DisparitySpaceWarp(const Matrix33& K, float b)
    : _K(K), _H(Matrix44::Identity())
{
  const auto fx = K(0,0), fy = K(1,1);

  _fx_i = 1.0f / fx;
  _fy_i = 1.0f / fy;
  _b_i = 1.0f / b;
  _bf_i = 1.0f / (b*fx);

  _G <<
      fx, 0, 0, 0,
      0, fy, 0, 0,
      0, 0, 0, fx*b,
      0, 0, 1, 0;

  _G_inv <<
      (1.0/fx), 0, 0, 0,
      0, (1.0/fy), 0, 0,
      0, 0, 0, 1,
      0, 0, (1.0/(fx*b)), 0;
}

} // bpvo
