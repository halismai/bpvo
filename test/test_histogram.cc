#include "bpvo/histogram.h"
#include <vector>

using namespace bpvo;

int main()
{
  int N = 1000;
  std::vector<float> data(N);
  for(int i = 0; i < N; ++i) {
    data[i] = 255.0 * (rand() / (float) RAND_MAX);
  }

  Histogram<float> hist(0, 255, 0.01);
  hist.add(data);

  printf("size %zu\n", hist.size());
  printf("median %f\n", hist.median());
  printf("true median %f\n", median(data));


  return 0;
}
