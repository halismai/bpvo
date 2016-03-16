#include "bpvo/approximate_median.h"
#include "bpvo/utils.h"

#include <vector>

int main()
{
  int N = 1000;
  std::vector<float> data(N);
  for(int i = 0; i < N; ++i) {
    data[i] = 255.0 * (rand() / (float) RAND_MAX);
  }

  auto data2(data);
  auto m_approx = bpvo::approximate_median(data2, 0.0, 255.0, 0.01);
  printf("approx: %f\n", m_approx);

  auto data1(data);
  auto m_true = bpvo::median(data1);
  printf("median: %f\n", m_true);

  return 0;
}
