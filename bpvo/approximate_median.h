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
#include <bpvo/histogram.h>

namespace bpvo {

template <class Iterator> inline typename Iterator::value_type
approximate_median(Iterator first, Iterator last,
                   typename Iterator::value_type min_val,
                   typename Iterator::value_type max_val,
                   typename Iterator::value_type resolution)
{
  static_assert( std::is_floating_point<typename Iterator::value_type>::value,
                "approximate_median is implemented for floating point only" );

  return Histogram<typename Iterator::value_type>(min_val, max_val, resolution).
      add(first, last).median();
}

template <class Iterator, class Func> inline typename Iterator::value_type
approximate_median(Iterator first, Iterator last,
                   typename Iterator::value_type min_val,
                   typename Iterator::value_type max_val,
                   typename Iterator::value_type resolution,
                   Func func)
{
  Histogram<typename Iterator::value_type> hist(min_val, max_val, resolution);
  for(auto it = first; it != last; ++it)
    hist.add( func(*it) );

  return hist.median();
}

template <class Container> inline typename Container::value_type
approximate_median(const Container& c,
                   typename Container::value_type min_val,
                   typename Container::value_type max_val,
                   typename Container::value_type res)
{
  return approximate_median(std::begin(c), std::end(c), min_val, max_val, res);
}

template <class Container, class Func> inline typename Container::value_type
approximate_median(const Container& c,
                   typename Container::value_type min_val,
                   typename Container::value_type max_val,
                   typename Container::value_type res,
                   Func f)
{
  return approximate_median(std::begin(c), std::end(c), min_val, max_val, res, f);
}


}; // bpvo

#endif // BPVO_APPROXIMATE_MEDIAN_H
