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

#ifndef BPVO_VECTOR6_SSE_H
#define BPVO_VECTOR6_SSE_H

namespace bpvo {

#if defined(__AVX__)
#include <immintrin.h>
#define V_LOAD( a ) _mm256_load_ps( (a) )
#define V_SET1( a ) _mm256_set1_ps( (a) )
#define V_STORE( dst, expr ) _mm256_store_ps( (dst), (expr) )
#define MAKE_OP( op_name ) _mm256_##op_name

#define MAKE_V_OP( op ) \
  Vector6 ret;          \
  V_STORE( ret._data, MAKE_OP( op ) ( V_LOAD(_data), V_LOAD(other._data) ) ); \
  return ret

#define MAKE_V_OP_E( op ) \
  V_STORE( _data, MAKE_OP( op ) ( V_LOAD(_data), V_LOAD(other._data) ) ); \
  return *this

#define MAKE_V_OP_S( op ) \
  Vector6 ret;            \
  V_STORE( ret._data, MAKE_OP( op ) ( V_LOAD(_data), V_SET1( x )) ); \
  return ret

#define MAKE_V_OP_S_E( op ) \
  V_STORE( _data, MAKE_OP( op ) ( V_LOAD(_data), V_SET1( x )) ); \
  return *this

inline constexpr int Vector6::ImplementationType() { return AVX; }

inline Vector6 floor(const Vector6& v)
{
  Vector6 ret;
  _mm256_store_ps(ret.data(), _mm256_floor_ps(_mm256_load_ps(v.data())));
  return ret;
}

inline Vector6 ceil(const Vector6& v)
{
  Vector6 ret;
  _mm256_store_ps(ret.data(), _mm256_ceil_ps(_mm256_load_ps(v.data())));
  return ret;
}

inline Vector6 round(const Vector6& v)
{
  Vector6 ret;
  _mm256_store_ps(ret.data(), _mm256_round_ps(_mm256_load_ps(v.data()), _MM_FROUND_TO_NEAREST_INT));
  return ret;
}

inline Vector6 abs(const Vector6& v)
{
  static const __m256 sign_mask = _mm256_set1_ps(-0.0f);
  Vector6 ret;
  _mm256_store_ps(ret.data(), _mm256_andnot_ps(sign_mask, _mm256_load_ps(v.data())));
  return ret;
}

#else

inline constexpr int Vector6::ImplementationType() { return SSE; }

inline Vector6 floor(const Vector6& v)
{
#if defined(__SSE4_1__)
  Vector6 ret;
  _mm_store_ps( ret.data() + 0, _mm_floor_ps( _mm_load_ps(v.data() + 0) ) );
  _mm_store_ps( ret.data() + 4, _mm_floor_ps( _mm_load_ps(v.data() + 4) ) );
  return ret;
#else
  return Vector6(
      std::floor( v[0] ),
      std::floor( v[1] ),
      std::floor( v[2] ),
      std::floor( v[3] ),
      std::floor( v[4] ),
      std::floor( v[5] ));
#endif
}

inline Vector6 ceil(const Vector6& v)
{
#if defined(__SSE4_1__)
  Vector6 ret;
  _mm_store_ps( ret.data() + 0, _mm_ceil_ps( _mm_load_ps(v.data() + 0) ) );
  _mm_store_ps( ret.data() + 4, _mm_ceil_ps( _mm_load_ps(v.data() + 4) ) );
  return ret;
#else
  return Vector6(
      std::ceil( v[0] ),
      std::ceil( v[1] ),
      std::ceil( v[2] ),
      std::ceil( v[3] ),
      std::ceil( v[4] ),
      std::ceil( v[5] ));
#endif
}

#define ROUNDING_MODE _MM_FROUND_TO_NEAREST_INT
inline Vector6 round(const Vector6& v)
{
#if defined(__SSE4_1__)
  Vector6 ret;
  _mm_store_ps( ret.data() + 0, _mm_round_ps( _mm_load_ps(v.data() + 0), ROUNDING_MODE ) );
  _mm_store_ps( ret.data() + 4, _mm_round_ps( _mm_load_ps(v.data() + 4), ROUNDING_MODE ) );
  return ret;
#else
  return Vector6(
      std::round( v[0] ),
      std::round( v[1] ),
      std::round( v[2] ),
      std::round( v[3] ),
      std::round( v[4] ),
      std::round( v[5] ));
#endif
}
#undef ROUNDING_MODE

inline Vector6 abs(const Vector6& v)
{
  static const __m128 sign_mask = _mm_set1_ps(-0.0f); // 1<<31

  Vector6 ret;
  _mm_store_ps( ret.data() + 0, _mm_andnot_ps(sign_mask, _mm_load_ps( v.data() + 0 )));
  _mm_store_ps( ret.data() + 4, _mm_andnot_ps(sign_mask, _mm_load_ps( v.data() + 4 )));

  return ret;
}

#define V_LOAD( a ) _mm_load_ps( (a) )
#define V_SET1( a ) _mm_set1_ps( (a) )
#define V_STORE( dst, expr ) _mm_store_ps( (dst), (expr) )
#define MAKE_OP( op_name ) _mm_##op_name

#define MAKE_V_OP( op ) \
  Vector6 ret;          \
  V_STORE( ret._data+0, MAKE_OP( op ) ( V_LOAD(_data+0), V_LOAD(other._data+0) ) ); \
  V_STORE( ret._data+4, MAKE_OP( op ) ( V_LOAD(_data+4), V_LOAD(other._data+4) ) ); \
  return ret

#define MAKE_V_OP_E( op ) \
  V_STORE( _data+0, MAKE_OP( op ) ( V_LOAD(_data+0), V_LOAD(other._data+0) ) ); \
  V_STORE( _data+4, MAKE_OP( op ) ( V_LOAD(_data+4), V_LOAD(other._data+4) ) ); \
  return *this


#define MAKE_V_OP_S( op ) \
  auto vv = V_SET1( x );  \
  Vector6 ret;            \
  V_STORE( ret._data+0, MAKE_OP( op ) ( V_LOAD(_data+0), vv) ); \
  V_STORE( ret._data+4, MAKE_OP( op ) ( V_LOAD(_data+4), vv) ); \
  return ret

#define MAKE_V_OP_S_E( op ) \
  auto vv = V_SET1( x );  \
  V_STORE( _data+0, MAKE_OP( op ) ( V_LOAD(_data+0), vv) ); \
  V_STORE( _data+4, MAKE_OP( op ) ( V_LOAD(_data+4), vv) ); \
  return *this

#endif // __AVX__

inline Vector6& Vector6::set(float v)
{
  V_STORE(_data, V_SET1(v));
  return *this;
}

inline Vector6& Vector6::set(const float* p)
{
#if defined(__AVX__)
  _mm256_store_ps(_data, _mm256_loadu_ps(p));
#else
  _mm_store_ps(_data + 0, _mm_loadu_ps(p));
  _data[4] = p[4];
  _data[5] = p[5];
#endif

  return *this;
}

inline Vector6& Vector6::set(float a0, float a1, float a2, float a3, float a4, float a5)
{
#if defined(__AVX__)
  _mm256_store_ps(_data, _mm256_setr_ps(a0, a1, a2, a3, a4, a5, 0.0f, 0.0f));
#else
  _mm_store_ps(_data + 0, _mm_setr_ps(a0, a1, a2, a3));
  _mm_store_ps(_data + 4, _mm_setr_ps(a4, a5, 0.0f, 0.0f));
#endif
  return *this;
}

inline Vector6 Vector6::operator*(const Vector6& other) const { MAKE_V_OP(mul_ps); }
inline Vector6 Vector6::operator+(const Vector6& other) const { MAKE_V_OP(add_ps); }
inline Vector6 Vector6::operator-(const Vector6& other) const { MAKE_V_OP(sub_ps); }
inline Vector6 Vector6::operator/(const Vector6& other) const { MAKE_V_OP(div_ps); }
inline Vector6 Vector6::operator^(const Vector6& other) const { MAKE_V_OP(xor_ps); }

inline Vector6 Vector6::operator*(const float& x) const { MAKE_V_OP_S(mul_ps); }
inline Vector6 Vector6::operator+(const float& x) const { MAKE_V_OP_S(add_ps); }
inline Vector6 Vector6::operator-(const float& x) const { MAKE_V_OP_S(sub_ps); }
inline Vector6 Vector6::operator/(const float& x) const { MAKE_V_OP_S(div_ps); }

inline Vector6& Vector6::operator*=(const Vector6& other) { MAKE_V_OP_E(mul_ps); }
inline Vector6& Vector6::operator+=(const Vector6& other) { MAKE_V_OP_E(add_ps); }
inline Vector6& Vector6::operator-=(const Vector6& other) { MAKE_V_OP_E(sub_ps); }
inline Vector6& Vector6::operator/=(const Vector6& other) { MAKE_V_OP_E(div_ps); }

inline Vector6& Vector6::operator*=(const float& x) { MAKE_V_OP_S_E(mul_ps); }
inline Vector6& Vector6::operator+=(const float& x) { MAKE_V_OP_S_E(add_ps); }
inline Vector6& Vector6::operator-=(const float& x) { MAKE_V_OP_S_E(sub_ps); }
inline Vector6& Vector6::operator/=(const float& x) { MAKE_V_OP_S_E(div_ps); }

inline Vector6 operator-(const Vector6& v)
{
  static const auto sign_mask = Vector6(-0.0f);
  return sign_mask ^ v;
}

inline void Vector6::RankUpdate(const Vector6& J, float w, float* data)
{
  __m128 wwww = _mm_set1_ps(w);
  __m128 v1234 = _mm_load_ps(J.data());
  __m128 v56xx = _mm_load_ps(J.data() + 4);

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
}


#undef V_LOAD
#undef V_SET1
#undef V_STORE
#undef MAKE_OP
#undef MAKE_V_OP
#undef MAKE_V_OP_E
#undef MAKE_V_OP_S
#undef MAKE_V_OP_S_E

} // bpvo

#endif // BPVO_VECTOR6_SSE_H
