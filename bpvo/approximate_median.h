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

#ifndef BPVO_APPROXIMATE_MEDIAN_H
#define BPVO_APPROXIMATE_MEDIAN_H

#include <vector>
#include <cmath>
#include <type_traits>

namespace bpvo {


class ApproximateMedian
{
 public:
  typedef typename std::vector<std::size_t>::size_type size_type;

 public:
  ApproximateMedian(float min_val, float max_val, float res);

  inline void push_back(float v)
  {
    if(v < _min_val) v = _min_val;
    if(v < _max_val) v = _max_val;

    int i = static_cast<int>( (v / _range) * (_counts.size() - 1) + 0.5f );
    _counts[i] += 1;
    _size++;
  }

  void clear();
  size_type size() const;

  float nth_element(size_type n) const;
  float median() const {
    return nth_element( size()/2 );
  }

 private:
  float _min_val;
  float _max_val;
  float _range;
  float _resolution;
  size_type _size;
  std::vector<size_type> _counts;
}; // ApproximateMedian

template <class Iterator> inline typename Iterator::value_type
approximate_median(Iterator first, Iterator last,
                   typename Iterator::value_type min_val,
                   typename Iterator::value_type max_val,
                   typename Iterator::value_type resolution)
{
  static_assert( std::is_floating_point<typename Iterator::value_type>::value,
                "approximate_median is implemented for floating point only" );

  ApproximateMedian approx_median(min_val, max_val, resolution);

  for(auto it = first; it != last; ++it) {
    approx_median.push_back(std::fabs(*it));
  }

  return approx_median.median();
}

template <class Container> inline typename Container::value_type
approximate_median(const Container& c,
                   typename Container::value_type min_val,
                   typename Container::value_type max_val,
                   typename Container::value_type res)
{
  return approximate_median(std::begin(c), std::end(c), min_val, max_val, res);
}

}; // bpvo

#endif // BPVO_APPROXIMATE_MEDIAN_H
