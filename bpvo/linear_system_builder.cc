#include "bpvo/linear_system_builder.h"
#include "bpvo/robust_loss.h"
#include "bpvo/utils.h"
#include "bpvo/debug.h"

#define LINEAR_SYSTEM_BUILDER_SERIAL 1

#if LINEAR_SYSTEM_BUILDER_SERIAL
#define TBB_PREVIEW_SERIAL_SUBSET 1
#endif

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#define WITH_GPL_CODE 1

namespace bpvo {

LinearSystemBuilder::LinearSystemBuilder(LossFunctionType loss_func)
    : _loss_func(loss_func) {}

template <class LossFunction>
struct LinearSystemBuilderReduction
{
  typedef typename LinearSystemBuilder::JacobianVector JacobianVector;
  typedef typename LinearSystemBuilder::ResidualsVector ResidualsVector;
  typedef std::vector<uint8_t> ValidVector;
  typedef typename LinearSystemBuilder::Hessian Hessian;
  typedef typename LinearSystemBuilder::Gradient Gradient;

  LinearSystemBuilderReduction(const JacobianVector& J, const ResidualsVector& residuals,
                               const ValidVector& valid, float sigma);

  LinearSystemBuilderReduction(LinearSystemBuilderReduction& other, tbb::split);

  ~LinearSystemBuilderReduction();

  void join(const LinearSystemBuilderReduction& other);

  void operator()(const tbb::blocked_range<int>& range);

  inline const Hessian& hessian() const { return _H; }
  inline const Gradient& gradient() const { return _G; }
  inline float residualsSquaredNorm() const { return _residuals_norm; }

 protected:
  const JacobianVector& _J;
  const ResidualsVector& _residuals;
  const ValidVector& _valid;
  LossFunction _loss_func;

  Hessian _H;
  Gradient _G;
  float _residuals_norm;

#if WITH_GPL_CODE
  Hessian toEigen(const float*) const;
#endif
}; // LinearSystemBuilderReduction

template <class RobustLoss> static inline
float RunSystemBuilder(const typename LinearSystemBuilder::JacobianVector& J,
                      const typename LinearSystemBuilder::ResidualsVector& R,
                      const std::vector<uint8_t>& valid, float sigma,
                      typename LinearSystemBuilder::Hessian& H,
                      typename LinearSystemBuilder::Gradient& g)
{
  LinearSystemBuilderReduction<RobustLoss> func(J, R, valid, sigma);
  tbb::blocked_range<int> range(0, (int) R.size());
  tbb::parallel_reduce(range, func);

  H = func.hessian();
  g = func.gradient();
  return func.residualsSquaredNorm();
}

static inline void getValidResiduals(const std::vector<uint8_t>& valid,
                                     const std::vector<float>& residuals,
                                     std::vector<float>& valid_residuals)
{
#define USE_ALL_DATA 0
  // residuals size is 8 times the size of valid
  // valid is stored per channel, so we loop 8 times!
  valid_residuals.clear();

#if USE_ALL_DATA
  valid_residuals.reserve(residuals.size());
  auto* ptr = residuals.data();

  for(int b = 0; b < 8; ++b) {
    for(size_t i=0; i < valid.size(); ++i, ++ptr)
      if(valid[i])
        valid_residuals.push_back(*ptr);
  }
#else
  // most of the other residuals will look the same. We'll cut the cost by
  // inspecting a single channel only
  valid_residuals.reserve(valid.size());
  for(size_t i = 0; i < valid.size(); ++i)
    if(valid[i])
      valid_residuals.push_back(residuals[i]);

#endif
#undef USE_ALL_DATA
}

static inline std::vector<uint8_t> makeValidFlags(const std::vector<uint8_t>& v)
{
  std::vector<uint8_t> ret(v.size()*8);
  auto* p = ret.data();

  for(int b = 0; b < 8; ++b)
    for(size_t i = 0; i < v.size(); ++i)
      *p++ = v[i];

  return ret;
}

float LinearSystemBuilder::run(const JacobianVector& J, const ResidualsVector& residuals,
                               const std::vector<uint8_t>& valid, Hessian& A, Gradient& b)
{
  assert( (J.size()-1) == residuals.size() && valid.size() == residuals.size()/8 );

  float scale = 1.0f;

  if(_loss_func != LossFunctionType::kL2) {
    // compute the std. deviation of the data using the valid points only
    getValidResiduals(valid, residuals, _tmp_buffer);
    scale = medianAbsoluteDeviation(_tmp_buffer) / 0.6745;
  }

  // for the case with zero residuals
  if(scale < 1e-6) scale = 1.0f;

  auto valid2 = makeValidFlags(valid);
  assert( valid2.size() == residuals.size() );

  float r_norm = 0.0;
  switch(_loss_func) {
    case LossFunctionType::kHuber:
      r_norm = RunSystemBuilder<Huber>(J, residuals, valid2, scale, A, b);
      break;
    case LossFunctionType::kTukey:
      r_norm = RunSystemBuilder<Tukey>(J, residuals, valid2, scale, A, b);
      break;
    case LossFunctionType::kL2:
      r_norm = RunSystemBuilder<L2Loss>(J, residuals, valid2, scale, A, b);
      break;
    default:
      THROW_ERROR("invalid robust loss");
  }

  return std::sqrt(r_norm);
}


float LinearSystemBuilder::run(const ResidualsVector& residuals,
                               const std::vector<uint8_t>& valid)
{
  float scale = 1.0f;
  if(_loss_func != LossFunctionType::L2) {
    getValidResiduals(valid, residuals, _tmp_buffer);
    scale = medianAbsoluteDeviation(_tmp_buffer) / 0.6745;
    if(scale < 1e-6)
      scale = 1.0f;
  }

  auto valid2 = makeValidFlags(valid);
  float ret = 0.0;

  for(size_t i = 0; i < valid2.size(); ++i) {
    float w = static_cast<float>(valid2[i]) * _loss_func.weight(residuals[i]);
    ret += w * _residuals[i] * _residuals[i];
  }

  return std::sqrt(ret);
}

template <class L> inline
LinearSystemBuilderReduction<L>::
LinearSystemBuilderReduction(const JacobianVector& J, const ResidualsVector& R,
                             const ValidVector& V, float sigma)
  : _J(J), _residuals(R), _valid(V), _loss_func(sigma)
  , _H(Hessian::Zero()), _G(Gradient::Zero()), _residuals_norm(0.0f)
{
  assert(_residuals.size() == _valid.size());
}

template <class L> inline
LinearSystemBuilderReduction<L>::
LinearSystemBuilderReduction(LinearSystemBuilderReduction& other, tbb::split)
  : _J(other._J), _residuals(other._residuals), _valid(other._valid), _loss_func(other._loss_func)
  , _H(Hessian::Zero()), _G(Gradient::Zero()), _residuals_norm(0.0f) {}

template <class L> inline
LinearSystemBuilderReduction<L>::~LinearSystemBuilderReduction() {}

template <class L> inline
void LinearSystemBuilderReduction<L>::join(const LinearSystemBuilderReduction& other)
{
  _H.noalias() += other._H;
  _G.noalias() += other._G;
  _residuals_norm += other._residuals_norm;
}

template <class L> inline
void LinearSystemBuilderReduction<L>::operator()(const tbb::blocked_range<int>& range)
{
  float r_norm = _residuals_norm;

#if WITH_GPL_CODE
  static constexpr int LENGTH = 24;
  alignas(16) float _data[LENGTH];

  memset(_data, 0, sizeof(float)*LENGTH);
#endif

  for(int i = range.begin(); i != range.end(); ++i) {
    if(_valid[i]) {
      float w = _loss_func.weight(_residuals[i]);
#if WITH_GPL_CODE
      __m128 wwww = _mm_set1_ps(w);
      __m128 v1234 = _mm_loadu_ps(_J[i].data());
      __m128 v56xx = _mm_loadu_ps(_J[i].data() + 4);

      __m128 v1212 = _mm_movelh_ps(v1234, v1234);
      __m128 v3434 = _mm_movehl_ps(v1234, v1234);
      __m128 v5656 = _mm_movelh_ps(v56xx, v56xx);

      __m128 v1122 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v1212, v1212));

      _mm_store_ps(_data + 0, _mm_add_ps(_mm_load_ps(_data + 0), _mm_mul_ps(v1122, v1212)));
      _mm_store_ps(_data + 4, _mm_add_ps(_mm_load_ps(_data + 4), _mm_mul_ps(v1122, v3434)));
      _mm_store_ps(_data + 8, _mm_add_ps(_mm_load_ps(_data + 8), _mm_mul_ps(v1122, v5656)));

      __m128 v3344 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v3434, v3434));

      _mm_store_ps(_data + 12, _mm_add_ps(_mm_load_ps(_data + 12), _mm_mul_ps(v3344, v3434)));
      _mm_store_ps(_data + 16, _mm_add_ps(_mm_load_ps(_data + 16), _mm_mul_ps(v3344, v5656)));

      __m128 v5566 = _mm_mul_ps(wwww, _mm_unpacklo_ps(v5656, v5656));
      _mm_store_ps(_data + 20, _mm_add_ps(_mm_load_ps(_data + 20), _mm_mul_ps(v5566, v5656)));

      _G.noalias() += _J[i].transpose() * _residuals[i] * w;
#else
      Gradient JwT = w * _J[i].transpose();
      _H.noalias() += JwT * _J[i];
      _G.noalias() += JwT * _residuals[i];
#endif
      r_norm += w * _residuals[i] * _residuals[i];
    }
  }

#if WITH_GPL_CODE
  _H.noalias() += toEigen(_data);
#endif

  _residuals_norm = r_norm;
}

#if WITH_GPL_CODE
template <class L> inline
auto LinearSystemBuilderReduction<L>::toEigen(const float* _data) const  -> Hessian
{
  Hessian ret;
  for(int i = 0, ii=0; i < 6; i += 2) {
    for(int j = i; j < 6; j += 2) {
      ret(i  , j  ) = _data[ii++];
      ret(i  , j+1) = _data[ii++];
      ret(i+1, j  ) = _data[ii++];
      ret(i+1, j+1) = _data[ii++];
    }
  }

  ret.template selfadjointView<Eigen::Upper>().evalTo(ret);
  return ret;
}
#endif

}; // bpvo

#if LINEAR_SYSTEM_BUILDER_SERIAL
#undef TBB_PREVIEW_SERIAL_SUBSET
#endif

#undef LINEAR_SYSTEM_BUILDER_SERIAL
#undef WITH_GPL_CODE

