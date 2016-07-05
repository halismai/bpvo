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


#include "bpvo/warps.h"
#include "bpvo/rigid_body_warp.h"

namespace bpvo {

RigidBodyWarp::RigidBodyWarp(const Matrix33& K, float b)
    : _K(K), _b(b), _T(Matrix44::Identity()), _T_inv(Matrix44::Identity()) {}

auto RigidBodyWarp::warpPoints(const PointVector& points) const -> ImagePointVector
{
  typedef Eigen::Matrix<float,4,Eigen::Dynamic> MatrixType;
  typedef Eigen::Map<const MatrixType, Eigen::Aligned> MapType;

  const auto N = points.size();
  Eigen::Matrix<float,3,Eigen::Dynamic> Xw = _P * MapType(points[0].data(), 4, N);

  ImagePointVector ret(N);
  for(size_t i = 0; i < N; ++i) {
    ret[i] = Xw.col(i).head<2>() * (1.0f / Xw(2,i));
  }

  return ret;
}

FORCE_INLINE __m128 div_ps(__m128 a, __m128 b)
{
#define USE_RCP 1

#if USE_RCP
  return _mm_mul_ps(a, _mm_rcp_ps(b));
#else
  return _mm_div_ps(a, b);
#endif

#undef USE_RCP
}

int RigidBodyWarp::computeJacobian(const PointVector& points, const float* IxIy, float* ret) const
{
  int N = points.size();
  Eigen::Matrix<float, Eigen::Dynamic, 6> J(N, 6);

  float fx = _K(0,0), fy = _K(1,1);
  float s = _T(0,0), c1 = _T_inv(0,3), c2 = _T_inv(1,3), c3 = _T_inv(2,3);

  const __m128 FX = _mm_set1_ps(fx);
  const __m128 FY = _mm_set1_ps(fy);
  const __m128 C1 = _mm_set1_ps(c1);
  const __m128 C2 = _mm_set1_ps(c2);
  const __m128 C3 = _mm_set1_ps(c3);
  const __m128 S = _mm_set1_ps(s);

  static const __m128 SIGN_MASK = _mm_set1_ps(-0.0);

  {
    {
      float* p = &J(0,0);
      const float* xyzw = points[0].data();
      const float* IxIy_p = IxIy;

      __m128 x, y, z, a, b, Ix, Iy, G1, G2;
      for(int i = 0; i <= N-4; i += 4, xyzw += 16, IxIy_p += 8, p += 4)
      {
        auto x1 = _mm_load_ps(xyzw +  0),
             x2 = _mm_load_ps(xyzw +  4),
             x3 = _mm_load_ps(xyzw +  8),
             x4 = _mm_load_ps(xyzw +  12);

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(0,0,0,0));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(0,0,0,0));
        x = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(1,1,1,1));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(1,1,1,1));
        y = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(2,2,2,2));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(2,2,2,2));
        z = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        G1 = _mm_load_ps(IxIy_p + 0);
        G2 = _mm_load_ps(IxIy_p + 4);
        Ix = _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(2,0,2,0));
        Iy = _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(3,1,3,1));

        Ix = _mm_mul_ps(FX, Ix);
        Iy = _mm_mul_ps(FY, Iy);

        //-(Iy*(Z-c3))/Z - 1.0/(Z*Z)*(Ix*X+Iy*Y)*(Y-c2);
        auto xIx_yIy = _mm_add_ps(_mm_mul_ps(x, Ix), _mm_mul_ps(y, Iy));
        xIx_yIy = _mm_mul_ps(xIx_yIy, _mm_sub_ps(y, C2));
        auto z2 = _mm_mul_ps(z,z);

        xIx_yIy = div_ps(xIx_yIy, z2);
        auto t1 = _mm_mul_ps(Iy, _mm_sub_ps(z, C3));
        auto t2 = div_ps(t1, z);

        t2 = _mm_xor_ps(t2, SIGN_MASK);
        auto t3 = _mm_sub_ps(t2, xIx_yIy);
        _mm_store_ps(p, t3);
        //p[i] = -1.0f/(Z*Z)*(Ix*X*fx+Iy*Y*fy)*(Y-c2)-(Iy*fy*(Z-c3))/Z;
      }
    }

    {
      float* p = &J(0,1);
      const float* xyzw = points[0].data();
      const float* IxIy_p = IxIy;
      __m128 x, y, z, a, b, Ix, Iy, G1, G2;
      for(int i = 0; i <= N-4; i += 4, xyzw += 16, IxIy_p += 8, p += 4)
      {
        auto x1 = _mm_load_ps(xyzw +  0),
             x2 = _mm_load_ps(xyzw +  4),
             x3 = _mm_load_ps(xyzw +  8),
             x4 = _mm_load_ps(xyzw +  12);

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(0,0,0,0));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(0,0,0,0));
        x = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(1,1,1,1));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(1,1,1,1));
        y = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(2,2,2,2));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(2,2,2,2));
        z = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        G1 = _mm_load_ps(IxIy_p + 0);
        G2 = _mm_load_ps(IxIy_p + 4);
        Ix = _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(2,0,2,0));
        Iy = _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(3,1,3,1));

        Ix = _mm_mul_ps(FX, Ix);
        Iy = _mm_mul_ps(FY, Iy);

        //t0 = (Ix*(Z-c3))/Z+1.0/(Z*Z)*(Ix*X+Iy*Y)*(X-c1);
        auto t0 = div_ps(_mm_mul_ps(Ix, _mm_sub_ps(z, C3)), z);
        auto t1 = _mm_add_ps(_mm_mul_ps(x, Ix), _mm_mul_ps(y, Iy));
        auto t2 = _mm_mul_ps(t1, _mm_sub_ps(x, C1));
        auto t3 = div_ps(t2, _mm_mul_ps(z,z));
        _mm_store_ps(p, _mm_add_ps(t0, t3));
      }
    }

    {
      float* p = &J(0,2);
      const float* xyzw = points[0].data();
      const float* IxIy_p = IxIy;
      __m128 x, y, z, a, b, Ix, Iy, G1, G2;
      for(int i = 0; i <= N-4; i += 4, xyzw += 16, IxIy_p += 8, p += 4)
      {
        auto x1 = _mm_load_ps(xyzw +  0),
             x2 = _mm_load_ps(xyzw +  4),
             x3 = _mm_load_ps(xyzw +  8),
             x4 = _mm_load_ps(xyzw +  12);

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(0,0,0,0));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(0,0,0,0));
        x = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(1,1,1,1));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(1,1,1,1));
        y = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(2,2,2,2));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(2,2,2,2));
        z = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        G1 = _mm_load_ps(IxIy_p + 0);
        G2 = _mm_load_ps(IxIy_p + 4);
        Ix = _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(2,0,2,0));
        Iy = _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(3,1,3,1));

        Ix = _mm_mul_ps(FX, Ix);
        Iy = _mm_mul_ps(FY, Iy);

        //t0 = (Iy*(X-c1))/Z-(Ix*(Y-c2))/Z;

        auto t0 = div_ps(_mm_sub_ps(_mm_mul_ps(Iy, _mm_sub_ps(x, C1)),
                                    _mm_mul_ps(Ix, _mm_sub_ps(y, C2))), z);
        _mm_store_ps(p, t0);
      }
    }

    {
      float* p = &J(0,3);
      const float* xyzw = points[0].data();
      const float* IxIy_p = IxIy;
      __m128 z, a, b, Ix, G1, G2;
      for(int i = 0; i <= N-4; i += 4, xyzw += 16, IxIy_p += 8, p += 4)
      {
        auto x1 = _mm_load_ps(xyzw +  0),
             x2 = _mm_load_ps(xyzw +  4),
             x3 = _mm_load_ps(xyzw +  8),
             x4 = _mm_load_ps(xyzw +  12);

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(2,2,2,2));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(2,2,2,2));
        z = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        G1 = _mm_load_ps(IxIy_p + 0);
        G2 = _mm_load_ps(IxIy_p + 4);
        Ix = _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(2,0,2,0));

        Ix = _mm_mul_ps(FX, Ix);

        //t0 = Ix/(Z*s);
        _mm_store_ps(p, div_ps(Ix, _mm_mul_ps(z, S)));
      }
    }

    {
      float* p = &J(0,4);
      const float* xyzw = points[0].data();
      const float* IxIy_p = IxIy;
      __m128 z, a, b, Iy, G1, G2;
      for(int i = 0; i <= N-4; i += 4, xyzw += 16, IxIy_p += 8, p += 4)
      {
        auto x1 = _mm_load_ps(xyzw +  0),
             x2 = _mm_load_ps(xyzw +  4),
             x3 = _mm_load_ps(xyzw +  8),
             x4 = _mm_load_ps(xyzw +  12);

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(2,2,2,2));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(2,2,2,2));
        z = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        G1 = _mm_load_ps(IxIy_p + 0);
        G2 = _mm_load_ps(IxIy_p + 4);
        Iy = _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(3,1,3,1));

        Iy = _mm_mul_ps(FY, Iy);
        // t0 = Iy/(Z*s);
        _mm_store_ps(p, div_ps(Iy, _mm_mul_ps(z, S)));
      }

    }

    {
      float* p = &J(0,5);
      const float* xyzw = points[0].data();
      const float* IxIy_p = IxIy;
      __m128 x, y, z, a, b, Ix, Iy, G1, G2;
      __m128 s_i = _mm_set1_ps(1.0 / s);
      for(int i = 0; i <= N-4; i += 4, xyzw += 16, IxIy_p += 8, p += 4)
      {
        auto x1 = _mm_load_ps(xyzw +  0),
             x2 = _mm_load_ps(xyzw +  4),
             x3 = _mm_load_ps(xyzw +  8),
             x4 = _mm_load_ps(xyzw +  12);

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(0,0,0,0));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(0,0,0,0));
        x = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(1,1,1,1));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(1,1,1,1));
        y = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        a = _mm_shuffle_ps(x1, x2, _MM_SHUFFLE(2,2,2,2));
        b = _mm_shuffle_ps(x3, x4, _MM_SHUFFLE(2,2,2,2));
        z = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));

        G1 = _mm_load_ps(IxIy_p + 0);
        G2 = _mm_load_ps(IxIy_p + 4);
        Ix = _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(2,0,2,0));
        Iy = _mm_shuffle_ps(G1, G2, _MM_SHUFFLE(3,1,3,1));

        Ix = _mm_mul_ps(FX, Ix);
        Iy = _mm_mul_ps(FY, Iy);

        //t0 = -(1.0/(Z*Z)*(Ix*X+Iy*Y))/s;

        auto t0 = _mm_mul_ps(s_i, _mm_add_ps(_mm_mul_ps(x, Ix), _mm_mul_ps(y, Iy)));
        auto t1 = div_ps(t0, _mm_mul_ps(z, z));
        _mm_store_ps(p, _mm_xor_ps(t1, SIGN_MASK));
      }
    }
  }

  Eigen::Matrix<float,6,Eigen::Dynamic> Jt = J.transpose();
  memcpy(ret, Jt.data(), 6*N*sizeof(float));

  /*
  int n_processed = 0;
  for( ; n_processed <= N-4; n_processed += 4)
    ;*/

  // this now is a multiple pf 16 always!
  int n_processed = N;
  return n_processed;
}

} // bpvo

