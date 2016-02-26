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

#ifndef BPVO_VECTOR6_H
#define BPVO_VECTOR6_H

#include <iosfwd>
#include <bpvo/types.h>

namespace bpvo {

class Vector6
{
 public:
  typedef Vector6 SelfType;
  typedef typename AlignedVector<SelfType>::type AlignedStdVector;

  enum {Scalar=0, SSE=16, AVX=32};
  static constexpr int ImplementationType();

 public:
  Vector6() {}

  Vector6(float v) { set(v); }

  Vector6(float a0, float a1, float a2, float a3, float a4, float a5) {
    set(a0, a1, a2, a3, a4, a5);
  }

  Vector6(const float* ptr) { set(ptr); }

  SelfType& set(float v);
  SelfType& set(float a0, float a1, float a2, float a3, float a4, float a5);
  SelfType& set(const float* ptr);

  SelfType& operator=(float v) { return set(v); }

  SelfType operator*(const SelfType&) const;
  SelfType operator+(const SelfType&) const;
  SelfType operator-(const SelfType&) const;
  SelfType operator/(const SelfType&) const;
  SelfType operator^(const SelfType&) const;

  SelfType operator*(const float&) const;
  SelfType operator+(const float&) const;
  SelfType operator-(const float&) const;
  SelfType operator/(const float&) const;

  // element wise
  SelfType& operator*=(const SelfType&);
  SelfType& operator+=(const SelfType&);
  SelfType& operator-=(const SelfType&);
  SelfType& operator/=(const SelfType&);

  SelfType& operator*=(const float&);
  SelfType& operator+=(const float&);
  SelfType& operator-=(const float&);
  SelfType& operator/=(const float&);

  inline const float& operator[](int i) const { return _data[i]; }
  inline       float& operator[](int i)       { return _data[i]; }

  inline const float* data() const { return _data; }
  inline       float* data()       { return _data; }

  friend std::ostream& operator<<(std::ostream&, const SelfType&v);

  static Vector6 Random();
  static inline Vector6 Zero() { return Vector6(0.0f); }
  static inline Vector6 Ones() { return Vector6(1.0f); }

  static void RankUpdate(const Vector6& J, float w, float* buf);

 protected:
  alignas(DefaultAlignment) float _data[8];
}; // Vector6

inline Vector6 operator*(float x, const Vector6& v) { return v * x; }
inline Vector6 operator+(float x, const Vector6& v) { return v + x; }
inline Vector6 operator-(float x, const Vector6& v) { return Vector6(x) - v; }
inline Vector6 operator/(float x, const Vector6& v) { return Vector6(x) / v; }

inline Vector6 floor (const Vector6&);
inline Vector6 ceil  (const Vector6&);
inline Vector6 round (const Vector6&);
inline Vector6 abs   (const Vector6&);
inline Vector6 operator-(const Vector6&);


}; // bpvo

#if defined(__SSE2__) || defined(__AVX__)
#include "bpvo/vector6-sse.h"
#else
#include "bpvo/vector6-scalar.h"
#endif

#endif // BPVO_VECTOR6_H
