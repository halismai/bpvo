#include "bpvo/mestimator.h"
#include "bpvo/math_utils.h"
#include "bpvo/utils.h"

#include <algorithm>

namespace bpvo {

template <typename T = float>
struct HuberOp
{
  inline HuberOp(T k = T(1.345)) : _k(k) { }

  inline T operator()(T r) const noexcept
  {
    T x = std::fabs(r);
    return (x < _k) ? 1.0f : (_k / x);
  }

  T _k;
}; // HuberOp

template <typename T = float>
struct TukeyOp
{
  inline TukeyOp(T t = T(4.685)) : _t(t), _t_inv(1.0 / t) {}

  inline T operator()(T r) const noexcept
  {
    T x = std::fabs(r);
    return (x < 1e-6) ? 1.0 : (x > _t) ? 0.0f : math::sq(1.0f - math::sq(_t_inv * x));
  }

  T _t, _t_inv;
}; // TukeyOp


template <class RobustFunction, class... Args> static inline
void computeWeights(const std::vector<float>& residuals, float sigma,
                    std::vector<float>& weights, Args ... args)
{
  RobustFunction robust_fn(std::forward<Args>(args)...);
  float sigma_inv = 1.0f / sigma;

  std::transform(residuals.begin(), residuals.end(), weights.begin(),
                 [=](float x) { return robust_fn(sigma_inv * x); });
}

template <class RobustFunction, class... Args> static inline
void computeWeights(const std::vector<float>& residuals, const std::vector<uint8_t>& valid,
                    float sigma, std::vector<float>& weights, Args ... args)
{
  RobustFunction robust_fn(std::forward<Args>(args)...);

  float sigma_inv = 1.0f / sigma;
  for(size_t i = 0; i < valid.size(); ++i) {
    weights[i] = valid[i] ? robust_fn(sigma_inv * residuals[i]) : 0.0f;
  }
}

void computeWeights(LossFunctionType loss_func, const std::vector<float>& residuals,
                    float sigma, std::vector<float>& weights)
{
  weights.resize(residuals.size());

  if(loss_func == LossFunctionType::kL2) {
    std::fill(weights.begin(), weights.end(), 1.0f);
    return;
  }

  switch(loss_func) {
    case LossFunctionType::kHuber: computeWeights<HuberOp<float>>(residuals, sigma, weights); break;
    case LossFunctionType::kTukey: computeWeights<TukeyOp<float>>(residuals, sigma, weights); break;
    default: THROW_ERROR("unkonwn RobustFunction");
  }
}

void computeWeights(LossFunctionType loss_func, const std::vector<float>& residuals,
                    const std::vector<uint8_t>& valid, float sigma, std::vector<float>& weights)
{
  assert( residuals.size() == valid.size() );
  weights.resize(valid.size());

  if(loss_func == LossFunctionType::kL2) {
    std::fill(weights.begin(), weights.end(), 1.0f);
    return;
  }

  switch(loss_func) {
    case LossFunctionType::kHuber: computeWeights<HuberOp<float>>(residuals, valid, sigma, weights); break;
    case LossFunctionType::kTukey: computeWeights<TukeyOp<float>>(residuals, valid, sigma, weights); break;
    default: THROW_ERROR("unkonwn RobustFunction");
  }
}

AutoScaleEstimator::AutoScaleEstimator(float t)
  : _scale(1.0), _delta_scale(1e10), _tol(t) {}

void AutoScaleEstimator::reset()
{
  _delta_scale = 1e10;
  _scale = 1.0;
}

float AutoScaleEstimator::getScale() const { return _scale; }

float AutoScaleEstimator::estimateScale(const std::vector<float>& residuals,
                                       const std::vector<uint8_t>& valid)
{
  assert( residuals.size() == valid.size() );
  if(_delta_scale > _tol) {
    _buffer.clear();

    for(size_t i = 0; i < residuals.size(); ++i) {
      if(valid[i])
        _buffer.push_back(std::fabs(residuals[i]));
    }

    auto z = 1.4826 * (1.0 + 5.0/(_buffer.size()-6));
    auto scale = z * median(_buffer);
    _delta_scale = std::fabs(scale - _scale);
    _scale = scale;
  }

  return _scale;
}

}; // bpvo
