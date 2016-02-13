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

#include "bpvo/approximate_median.h"

#include <cassert>

namespace bpvo {

ApproximateMedian::ApproximateMedian(float min_val, float max_val, float res)
    : _min_val(min_val), _max_val(max_val), _range(std::abs(_max_val-_min_val))
      , _resolution(res), _size(0)
{
  assert( _max_val > _min_val );

  size_type n = static_cast<size_type>( _range / _resolution );

  _counts.resize(n, 0);
}

static inline float clamp(float v, float min_val, float max_val)
{
  if(v < min_val) v = min_val;
  if(v > max_val) v = max_val;

  return v;
}

/*
void ApproximateMedian::push_back(float v)
{
  v = clamp(v, _min_val, _max_val);
  int i = static_cast<int>( (v / _range) * (_counts.size() - 1) + 0.5f );
  _counts[i] += 1;
  _size++;
}
*/

float ApproximateMedian::nth_element(size_type nth) const
{
  size_type i = 0;
  size_type n = _counts[i++];

  while(n < nth) {
    n += _counts[i++];
  }
  auto n_prev = n - _counts[--i];
  if(i) --i;

  float f = (nth - n_prev) / static_cast<float>(n - n_prev);
  return (i + f) * _resolution;
}

void ApproximateMedian::clear()
{
  std::fill(_counts.begin(), _counts.end(), 0.0f);
  _size = 0;
}

auto ApproximateMedian::size() const -> size_type
{
  return _size;
}

}; // bpvo

