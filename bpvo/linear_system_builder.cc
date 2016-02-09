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

#include "bpvo/linear_system_builder.h"
#include "bpvo/linear_system_builder_reduction.h"

namespace bpvo {

static inline void getValidResiduals(const std::vector<uint8_t>& valid,
                                     const std::vector<float>& residuals,
                                     std::vector<float>& valid_residuals)
{
#define USE_ALL_DATA 0
  // residuals size is 8 times the size of valid
  // valid is stored per channel, so we loop 8 times!
  valid_residuals.clear();

#if USE_ALL_DATA
  valid_residuals.reserve(residuals.size());
  auto* ptr = residuals.data();

  for(int b = 0; b < 8; ++b) {
    for(size_t i=0; i < valid.size(); ++i, ++ptr)
      if(valid[i])
        valid_residuals.push_back(*ptr);
  }
#else
  // most of the other residuals will look the same. We'll cut the cost by
  // inspecting a single channel only
  valid_residuals.reserve(valid.size());
  for(size_t i = 0; i < valid.size(); ++i)
    if(valid[i])
      valid_residuals.push_back(residuals[i]);

#endif
#undef USE_ALL_DATA
}

/**
 * makes the valid vector flags the same size as the number of residuals
 * NC is the number of channels of the descriptor
 */
static inline
std::vector<uint8_t> makeValidFlags(const std::vector<uint8_t>& v, int NC = 8)
{
  std::vector<uint8_t> ret(v.size()*NC);
  auto* p = ret.data();

  for(int b = 0; b < NC; ++b)
    for(size_t i = 0; i < v.size(); ++i)
      *p++ = v[i];

  return ret;
}

float LinearSystemBuilder::Run(const JacobianVector& J, const ResidualsVector& residuals,
                               const ResidualsVector& weights, const std::vector<uint8_t>& valid,
                               Hessian* A, Gradient* b)
{
  auto nc = residuals.size() / valid.size();
  assert( (J.size()-1) == residuals.size() && valid.size() == residuals.size()/nc );

  auto valid2 = residuals.size() != valid.size() ? makeValidFlags(valid, nc) : valid;
  assert( valid2.size() == residuals.size() );

  float res_sq_norm=LinearSystemBuilderReduction::Run(J, residuals, weights, valid, A, b);
  return std::sqrt(res_sq_norm);
}

}; // bpvo

