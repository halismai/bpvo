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

#ifndef BPVO_HISTOGRAM_H
#define BPVO_HISTOGRAM_H

#include <opencv2/core/core.hpp>
#include <algorithm>

#include <bpvo/utils.h>
#include <bpvo/math_utils.h>

namespace bpvo {

template <typename T = float>
class Histogram
{
 public:
  /**
   * creates a histogram
   *
   * \param min_val minimum value that histogram will handle
   * \param max_val maximum value
   * \param resolution, i.e. the bucket size
   */
  Histogram(T min_val, T max_val, T resolution)
      : _min_val(min_val)
      , _max_val(max_val)
      , _resolution(resolution)
      , _range(_max_val - _min_val)
      , _size(static_cast<size_t>(std::ceil(_range / _resolution)))
      , _samples(0)
      , _counts(_size)
  {
    clear();
  }

  inline void clear()
  {
    std::fill_n(&_counts[0], _size, 0);
    _samples = 0;
  }

  inline void add(const T& v_)
  {
    const T v = std::max(_min_val, std::min(v_, _max_val));
    const T f = (v / _range) * (_size - 1);
    const int i = math::Floor( f );
    ++_counts[i];
    ++_samples;
  }

  template <class Iterator> inline
  const Histogram<T>& add(Iterator first, Iterator last)
  {
    for(auto it = first; it != last; ++it)
      add(*it);

    return *this;
  }

  template <class Container> inline
  const Histogram<T>& add(const Container& data)
  {
    return add(std::begin(data), std::end(data));
  }

  inline T nthElement(size_t nth) const
  {
    size_t sum = 0, bin = 0;
    for(bin = 0; sum < nth; ++bin )
      sum += _counts[bin];

    size_t prev = sum - _counts[bin > 0 ? bin-1 : bin];
    float f = (nth - prev) / (float) (sum - prev);
    float ret = (bin + f) * _resolution;

    return ret;
  }

  inline T median() const
  {
    return nthElement( (_samples + 1) / 2 );
  }

  inline size_t numSamples() const { return _samples; }
  inline size_t size() const { return _size; }

 private:
  T _min_val;
  T _max_val;
  T _resolution;
  T _range;
  size_t _size;
  size_t _samples;
  cv::AutoBuffer<size_t,5100> _counts;
}; // Histogram

}; // bpvo

#endif // BPVO_HISTOGRAM_H
