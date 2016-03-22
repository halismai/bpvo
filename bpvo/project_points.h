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

#ifndef BPVO_WARP_POINTS_H
#define BPVO_WARP_POINTS_H

#include <bpvo/types.h>

namespace bpvo {

/**
 * Project 3D points on the camera given pose
 *
 * \param P is 3x4 extrinsic camera matrix P = K*[R t]
 * \param X pointer to 3D points represented with homogenous coordinates [x,y,z,1]
 * \param N the number of points
 * \param uv projections onto the image [u_i, v_i]
 * \param image_size the image size to determine the valid points
 * \param valid is set to nonzero value if the point projects onto the image
 */
void projectPoints(const Matrix34& P, const float* xyzw, int N, float* uv,
                   const ImageSize& image_size, typename ValidVector::value_type* valid);

void projectPoints(const Matrix34& P, const float* xyzw, int N, float* uv,
                   const ImageSize& image_size, typename ValidVector::value_type* valid, int* inds);

void projectPoints(const Matrix34& P, const float* xyzw, int N,
                   const ImageSize& image_size, typename ValidVector::value_type* valid,
                   int* inds, float* C);



}; // bpvo

#endif // BPVO_WARP_POINTS_H
