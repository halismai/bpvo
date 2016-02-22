#include <bpvo/config.h>
#include <vector>

#if defined(WITH_TBB)
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

int main()
{
  std::vector<int> v(100);
#if 0 && defined(WITH_OPENMP)
#pragma omp parallel for
#endif
  for(int i = 0; i < (int) v.size(); ++i)
  {
    v[i] = i;
  }

#if defined(WITH_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(0, (int) v.size()),
                    [&](const tbb::blocked_range<int>& r)
                    {
                      for(int i = r.begin(); i != r.end(); ++i) v[i] = i;
                    });
#endif

  return 0;
}
