#include "bpvo/linear_system_builder_2.h"
#include "bpvo/utils.h"

namespace bpvo {

LinearSystemBuilder2::LinearSystemBuilder2(LossFunctionType l_type)
    : _loss_func_type(l_type) {}

static inline float SQ(float x) { return x*x; }

float estimateErrorScale(const std::vector<float>& R, const std::vector<uint8_t>& valid)
{
  std::vector<float> tmp;
  tmp.reserve(R.size());

  for(size_t i = 0; i < valid.size(); ++i)
    if(valid[i])
      tmp.push_back(std::fabs(R[i]));

  float ret = 1.4826f * median(tmp);
  if(ret < 1e-6)
    ret = 1.0f;
  return ret;
}

std::vector<float> computeWeights(LossFunctionType loss_fn_type,
                                  const float sigma_inv,
                                  const std::vector<uint8_t>& V,
                                  const std::vector<float>& R)
{
  std::vector<float> weights;
  weights.resize(V.size());

  for(size_t i = 0; i < weights.size(); ++i)
  {
    weights[i] = 0.0f;
    if(V[i])
    {
      float r = std::fabs( R[i] * sigma_inv );

      switch(loss_fn_type) {
        case kHuber: weights[i] = r < 1.345f ? 1.0f : (1.345f / r); break;
        case kTukey: weights[i] = r > 4.685f ? 0.0f : SQ(1.0f - SQ(r/4.685f)); break;
        case kL2: weights[i] = 1.0f; break;
        default: THROW_ERROR("invalid LossFunctionType\n");
      }
    }
  }

  return weights;
}

float LinearSystemBuilder2::run(const JacobianVector& J, const ResidualsVector& R,
                                const ValidVector& V, Hessian& H, Gradient& b)
{
  assert( J.size()-1 == V.size() );
  assert( R.size() == V.size() );

  float sigma = estimateErrorScale(R, V);
  auto weights = computeWeights(_loss_func_type, 1.0f/sigma, V, R);

  H.setZero();
  b.setZero();

  float ret = 0.0f;
  for(size_t i = 0; i < V.size(); ++i)
  {
    if(V[i])
    {
      H.noalias() += weights[i] * J[i].transpose() * J[i];
      b.noalias() += weights[i] * R[i] * J[i].transpose();
      ret += weights[i] * R[i] * R[i];
    }
  }

  return ret;
}

float LinearSystemBuilder2::run(const ResidualsVector& R, const ValidVector& V)
{
  assert( R.size() == V.size() );

  float sigma = estimateErrorScale(R, V);
  auto weights = computeWeights(_loss_func_type, 1.0f/sigma, V, R);

  float ret = 0.0f;
  for(size_t i = 0; i < V.size(); ++i)
    ret += weights[i] * R[i] * R[i];

  return ret;
}

}; // bpvo
