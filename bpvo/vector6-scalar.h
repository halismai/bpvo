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

#ifndef BPVO_VECTOR6_SCALAR_H
#define BPVO_VECTOR6_SCALAR_H

#include <cmath>

namespace bpvo {

inline constexpr int Vector6::ImplementationType() { return Scalar; }

inline Vector6& Vector6::set(float v)
{
  for(int i = 0; i < 6; ++i)
    _data[i] = v;

  _data[6] = _data[7] = 0.0f;
  return *this;
}

inline Vector6& Vector6::set(float a0, float a1, float a2, float a3, float a4, float a5)
{
  _data[0] = a0;
  _data[1] = a1;
  _data[2] = a2;
  _data[3] = a3;
  _data[4] = a4;
  _data[5] = a5;
  _data[6] = _data[7] = 0.0f;

  return *this;
}

inline Vector6& Vector6::set(const float* p)
{
  for(int i = 0; i < 6; ++i)
    _data[i] = p[i];

  _data[6] = _data[7] = 0.0f;
  return *this;
}

#define MAKE_SCALAR_OP( op ) \
    Vector6 ret;             \
    ret._data[0] = _data[0] op other._data[0]; \
    ret._data[1] = _data[1] op other._data[1]; \
    ret._data[2] = _data[2] op other._data[2]; \
    ret._data[3] = _data[3] op other._data[3]; \
    ret._data[4] = _data[4] op other._data[4]; \
    ret._data[5] = _data[5] op other._data[5]; \
    return ret


inline Vector6 Vector6::operator*(const SelfType& other) const { MAKE_SCALAR_OP(*); }
inline Vector6 Vector6::operator+(const SelfType& other) const { MAKE_SCALAR_OP(+); }
inline Vector6 Vector6::operator/(const SelfType& other) const { MAKE_SCALAR_OP(/); }
inline Vector6 Vector6::operator-(const SelfType& other) const { MAKE_SCALAR_OP(-); }

#undef MAKE_SCALAR_OP

inline Vector6 Vector6::operator^(const Vector6& other) const
{
  Vector6 ret;
  ret[0] = static_cast<unsigned>( _data[0] ) ^ static_cast<unsigned>( other._data[0] );
  ret[1] = static_cast<unsigned>( _data[1] ) ^ static_cast<unsigned>( other._data[1] );
  ret[2] = static_cast<unsigned>( _data[2] ) ^ static_cast<unsigned>( other._data[2] );
  ret[3] = static_cast<unsigned>( _data[3] ) ^ static_cast<unsigned>( other._data[3] );
  ret[4] = static_cast<unsigned>( _data[4] ) ^ static_cast<unsigned>( other._data[4] );
  ret[5] = static_cast<unsigned>( _data[5] ) ^ static_cast<unsigned>( other._data[5] );
  return ret;
}

#define MAKE_SCALAR_OP( op ) \
    Vector6 ret;              \
    ret._data[0] = _data[0] op x; \
    ret._data[1] = _data[1] op x; \
    ret._data[2] = _data[2] op x; \
    ret._data[3] = _data[3] op x; \
    ret._data[4] = _data[4] op x; \
    ret._data[5] = _data[5] op x; \
    return ret;

inline Vector6 Vector6::operator*(const float& x) const { MAKE_SCALAR_OP(*); }
inline Vector6 Vector6::operator+(const float& x) const { MAKE_SCALAR_OP(+); }
inline Vector6 Vector6::operator-(const float& x) const { MAKE_SCALAR_OP(-); }
inline Vector6 Vector6::operator/(const float& x) const { MAKE_SCALAR_OP(/); }

#undef MAKE_SCALAR_OP

#define MAKE_SCALAR_OP( op ) \
    _data[0] = _data[0] op other._data[0]; \
    _data[1] = _data[1] op other._data[1]; \
    _data[2] = _data[2] op other._data[2]; \
    _data[3] = _data[3] op other._data[3]; \
    _data[4] = _data[4] op other._data[4]; \
    _data[5] = _data[5] op other._data[5]; \
    return *this

inline Vector6& Vector6::operator*=(const SelfType& other) { MAKE_SCALAR_OP(*); }
inline Vector6& Vector6::operator+=(const SelfType& other) { MAKE_SCALAR_OP(+); }
inline Vector6& Vector6::operator-=(const SelfType& other) { MAKE_SCALAR_OP(-); }
inline Vector6& Vector6::operator/=(const SelfType& other) { MAKE_SCALAR_OP(/); }

#undef MAKE_SCALAR_OP

#define MAKE_SCALAR_OP( op ) \
    _data[0] = _data[0] op x; \
    _data[1] = _data[1] op x; \
    _data[2] = _data[2] op x; \
    _data[3] = _data[3] op x; \
    _data[4] = _data[4] op x; \
    _data[5] = _data[5] op x; \
    return *this;

inline Vector6& Vector6::operator*=(const float& x) { MAKE_SCALAR_OP(*); }
inline Vector6& Vector6::operator+=(const float& x) { MAKE_SCALAR_OP(+); }
inline Vector6& Vector6::operator-=(const float& x) { MAKE_SCALAR_OP(-); }
inline Vector6& Vector6::operator/=(const float& x) { MAKE_SCALAR_OP(/); }

inline Vector6 Vector6::floor(const Vector6& v)
{
  return Vector6(
      std::floor( v[0] ),
      std::floor( v[1] ),
      std::floor( v[2] ),
      std::floor( v[3] ),
      std::floor( v[4] ),
      std::floor( v[5] ));
}

inline Vector6 Vector6::ceil(const Vector6& v)
{
  return Vector6(
      std::ceil( v[0] ),
      std::ceil( v[1] ),
      std::ceil( v[2] ),
      std::ceil( v[3] ),
      std::ceil( v[4] ),
      std::ceil( v[5] ));
}

inline Vector6 Vector6::round(const Vector6& v)
{
  return Vector6(
      std::round( v[0] ),
      std::round( v[1] ),
      std::round( v[2] ),
      std::round( v[3] ),
      std::round( v[4] ),
      std::round( v[5] ));
}

inline Vector6 Vector6::abs(const Vector6& v)
{
  return Vector6(
      std::abs( v[0] ),
      std::abs( v[1] ),
      std::abs( v[2] ),
      std::abs( v[3] ),
      std::abs( v[4] ),
      std::abs( v[5] ));
}

inline Vector6 Vector6::operator-(const Vector6& v)
{
  return Vector6(-v[0], -v[1], -v[2], -v[3], -v[4], -v[5]);
}

} // bpvo

#endif // BPVO_VECTOR6_SCALAR_H
