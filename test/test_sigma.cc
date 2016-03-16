#include "bpvo/utils.h"

#include <vector>
#include <random>

float MadEstimate(const std::vector<float>& f)
{
  std::vector<float> f_copy(f);
  return bpvo::medianAbsoluteDeviation(f_copy) / 0.6745;
}

float RobustStdDevEstimate(const std::vector<float>& f)
{
  std::vector<float> f_copy(f.size());
  for(size_t i = 0; i < f.size(); ++i)
    f_copy[i] = std::abs(f[i]);

  float z = 1.4826 * (1.0 + 5.0 / (f.size() - 6));
  return z * bpvo::median(f_copy.begin(), f_copy.end());
}

int main()
{
  std::random_device rd;
  std::mt19937 gen(rd());

  double m0 = 100.0,
         s0 = 5.0;
  std::normal_distribution<float> dist(m0, s0);

  int N = 1000;
  std::vector<float> data(N);
  for(int i = 0; i < N; ++i) {
    data[i] = dist(gen);
  }

  auto s1 = (double) MadEstimate(data),
       s2 = (double) RobustStdDevEstimate(data);

  printf("%f (%f)  %f (%f)\n", s1, std::abs(s1-s0), s2, std::abs(s2-s0));

  return 0;
}
