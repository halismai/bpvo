#include "bpvo/robust_fit.h"
#include <random>

using namespace bpvo;
ResidualsVector GetData(int N, float m, float s)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(m, s);

  ResidualsVector ret(N);
  for(int i = 0; i < N; ++i)
    ret[i] = dist(gen);

  return ret;
}

int main()
{
  float m0 = 100.0;
  float s0 = 5.0;
  int N = 10000;
  auto data = GetData(N, m0, s0);

  RobustFit robust_fit(LossFunctionType::kHuber);

  ValidVector valid(data.size(), 1);
  printf("sigma: %f\n", robust_fit.estimateScale(data, valid));

  return 0;
}

