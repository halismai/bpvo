#include "bpvo/robust_fit.h"
#include "bpvo/utils.h"

namespace bpvo {

static inline float GetDefaultTuningConstant(LossFunctionType loss)
{
  switch(loss)
  {
    case LossFunctionType::kHuber: return 1.345f; break;
    case LossFunctionType::kL2: return 1.0f; break;
    case LossFunctionType::kTukey: return 4.685f; break;
    default: THROW_ERROR("unknown LossFunctionType\n");
  }
}

RobustFit::RobustFit(LossFunctionType loss, float t)
    : _loss(loss), _tuning_constant( t > 0.0 ? t : GetDefaultTuningConstant(loss) )
 {}

float RobustFit::estimateScale(const ResidualsVector& r, const ValidVector& v)
{
  if(_loss == LossFunctionType::kL2)
    return 1.0;

  _buffer.resize(0);
  _buffer.reserve(r.size());
  THROW_ERROR_IF( r.size() != v.size(), "size mismatch" );

  for(size_t i = 0; i < r.size(); ++i)
    if(v[i]) _buffer.push_back(std::fabs(r[i]));

  return median(_buffer) / 0.6745f;
}

void RobustFit::computeWeights(const ResidualsVector& r, const ValidVector& v,
                               WeightsVector& w, float scale)
{
  if(scale < 0.0f)
    scale = estimateScale(r, v);

  float s = _tuning_constant / scale;

  w.resize( r.size() );
  for(size_t i = 0; i < r.size(); ++i)
    w[i] = static_cast<float>(v[i]) * weight(s * r[i]);
}

static inline float square(float v) { return v*v; }

float RobustFit::weight(float r) const
{
  float w = 1.0f;

  switch(_loss)
  {
    case LossFunctionType::kHuber:
      w = 1.0f / std::max(1.0f, std::fabs(r));
      break;

    case LossFunctionType::kTukey:
      w = (std::fabs(r) < 1.0f) * square(1.0 - square(r));

    case LossFunctionType::kL2:
      w = 1.0f;
  }

  return w;
}

}; // bpvo


