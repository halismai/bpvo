#ifndef BPVO_STL_H
#define BPVO_STL_H

#include <algorithm>

namespace bpvo {

template <typename Iterator> inline
double sum(Iterator first, Iterator last)
{
  return std::accumulate(first, last, 0.0);
}

template <typename Iterator> inline
double mean(Iterator first, Iterator last)
{
  return sum(first, last) / (double) std::distance(first, last);
}

}; // bpvo

#endif // BPVO_STL_H
