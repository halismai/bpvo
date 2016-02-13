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

  auto m_true = bpvo::median(data);
  printf("median: %f\n", m_true);

  auto m_approx = bpvo::approximate_median(data, 0.0, 255.0, 0.25);
  printf("approx: %f\n", m_approx);

  return 0;
}
