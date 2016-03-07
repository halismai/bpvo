#include <bpvo/parallel.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>

using namespace bpvo;

template <typename T>
class Body : public ParallelForBody
{
 public:
  Body(std::vector<T>& v) : _v(v) {}

  inline void operator()(const Range& range) const
  {
    auto s = std::begin(_v) + range.begin(),
         e = s + range.size();
    std::iota(s, e, static_cast<T>(range.begin()));
  }

 protected:
  std::vector<T>& _v;
}; // Body

int main()
{
  printf("num threads %d\n", getNumMaxThreads());
  printf("num CPU's %d\n", getNumberOfCPUs());

  std::vector<int> v(100, -1);
  Body<int> body(v);

  parallel_for(Range(0, v.size()), body);
  parallel_for(Range(0, 1), body);

  return 0;
}

