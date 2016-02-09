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

#ifndef BPVO_ROBUST_LOSS_H
#define BPVO_ROBUST_LOSS_H

#include <bpvo/debug.h>

namespace bpvo {

namespace {
template <typename T> static FORCE_INLINE T square(T x) { return x*x; }
}

struct RobustLossFunction
{
  float _s_inv;

  RobustLossFunction(float sigma, float tune) noexcept
      : _s_inv( 1.0f / (sigma * tune) ) {}

  RobustLossFunction(const RobustLossFunction& o) noexcept
      : _s_inv( o._s_inv ) {}

  FORCE_INLINE float normalize(float x) const noexcept
  {
    return _s_inv * x;
  }

  FORCE_INLINE float normalizer() const noexcept { return _s_inv; }
}; // RobustLossFunction

struct Huber : public RobustLossFunction
{
  Huber(float sigma, float tune = 1.345) noexcept : RobustLossFunction(sigma, tune) {}

  FORCE_INLINE float weight(float x) const noexcept
  {
    float r = RobustLossFunction::normalize(x);
    return 1.0f / std::max(1.0f, std::fabs(r));
  }
}; // Huber

struct Tukey : public RobustLossFunction
{
  Tukey(float sigma, float tune = 4.685) noexcept : RobustLossFunction(sigma, tune) {}

  FORCE_INLINE float weight(float x) const noexcept
  {
    auto r = std::fabs(RobustLossFunction::normalize(x));
    return r < 1.0f ? square(1.0f - square(r)) : 0.0f;
  }
}; // Tukey

struct L2Loss : public RobustLossFunction
{
  L2Loss(float sigma = 1.0f, float tune = 1.0f) noexcept : RobustLossFunction(sigma, tune) {}

  FORCE_INLINE float weight(float) const noexcept { return 1.0f; }
}; // L2Loss

struct Andrews : public RobustLossFunction
{
  static constexpr float PI = 3.14159265359f;

  Andrews(float sigma, float tune = 1.339) noexcept : RobustLossFunction(sigma, tune) {}

  FORCE_INLINE float weight(float x) const noexcept
  {
    auto r = RobustLossFunction::normalize(x);
    return std::fabs(r) < PI ? std::sin(r) / r : 0.0f;
  }
}; // Andrews

struct Cauchy : public RobustLossFunction
{
  Cauchy(float sigma, float tune = 2.385)  noexcept : RobustLossFunction(sigma, tune) {}

  FORCE_INLINE float weight(float x) const noexcept
  {
    auto r = RobustLossFunction::normalize(x);
    return 1.0f / (1.0f + square(r));
  }
}; // Caucy

struct Fair : public RobustLossFunction
{
  Fair(float sigma, float tune = 1.4) noexcept : RobustLossFunction(sigma, tune) {}

  FORCE_INLINE float weight(float x) const noexcept
  {
    auto r = RobustLossFunction::normalize(x);
    return 1.0f / (1.0f + std::fabs(r));
  }
}; // Fair

}; // bpvo

#endif // BPVO_ROBUST_LOSS_H

