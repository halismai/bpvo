#include "bpvo/linear_system_builder.h"
//#include "bpvo/robust_loss.h"
#include "bpvo/utils.h"
#include "bpvo/debug.h"
#include "bpvo/math_utils.h"
#include "bpvo/utils.h"

#define LINEAR_SYSTEM_BUILDER_SERIAL 1

#if LINEAR_SYSTEM_BUILDER_SERIAL
#define TBB_PREVIEW_SERIAL_SUBSET 1
#endif

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/mutex.h>

#include <iostream>
#include <limits>

#define WITH_GPL_CODE 0
#define DEBUG_THE_STUFF 0

namespace bpvo {

template <typename T = float>
struct L2Op
{
  inline L2Op(T = 0.0) {}

  inline T operator()(T) const { return 1.0f; }
};

template <typename T = float>
struct HuberOp
{
  inline HuberOp(T k = T(1.345)) : _k(k) { }

  inline T operator()(T r) const
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

  inline T operator()(T r) const
  {
    T x = std::fabs(r);
    return (x < 1e-6) ? 1.0 : (x > _t) ? 0.0f : math::sq(1.0f - math::sq(_t_inv * x));
  }

  T _t, _t_inv;
}; // TukeyOp

template <typename T = float>
struct L1Op
{
  inline L1Op(T = 0.0) {}

  inline T operator()(T x) const
  {
    return T(1.0) / std::abs(x);
  }
}; // L1Op

template <typename T = float>
struct CauchyOp
{
  inline CauchyOp(T c = 2.3849) : _c_inv( T(1.0) / c ) {}

  inline T operator()(T x) const
  {
    return T(1) / (T(1) + math::sq(_c_inv * x));
  }

  T _c_inv;
}; // Cauchy

template <typename T = float>
struct GemanMcClure
{
  GemanMcClure(T = T(0)) {}

  inline T operator()(T x) const
  {
    return T(1) / math::sq(T(1) + math::sq(x));
  }
}; // GemanMcClure

template <typename T = float>
struct Welsch
{
  inline Welsch(T c = 2.9846) : _c_inv(T(1) / c) {}

  inline T operator()(T x) const
  {
    return std::exp( -math::sq(_c_inv*x) );
  }

  T _c_inv;
}; // Welsch


template <class Container> static inline typename
Container::value_type computeRobustStandarDeviation(Container& residuals, int p = 6)
{
#pragma omp simd
  for(size_t i = 0; i < residuals.size(); ++i)
    residuals[i] = std::abs(residuals[i]);

  return 1.4826f * (1.0 + 5.0/(residuals.size()-p)) * median(residuals);
}

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
  float _sigma_inv;

  Hessian _H = Hessian::Zero();
  Gradient _G = Gradient::Zero();
  float _residuals_norm = 0.0f;

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

  // XXX it is not always 8 channels
  for(int b = 0; b < 8; ++b)
    for(size_t i = 0; i < v.size(); ++i)
      *p++ = v[i];

  return ret;
}

LinearSystemBuilder::LinearSystemBuilder(LossFunctionType loss_func)
    : _loss_func(loss_func), _sigma(1.0), _delta_sigma(std::numeric_limits<float>::max()) {}

float LinearSystemBuilder::run(const JacobianVector& J, const ResidualsVector& residuals,
                               const std::vector<uint8_t>& valid, Hessian& A, Gradient& b)
{
  //assert( (J.size()-1) == residuals.size() && valid.size() == residuals.size()/8 );
  assert( (J.size()-1) == residuals.size() && valid.size() == residuals.size() );

  if(_loss_func != LossFunctionType::kL2 && _delta_sigma > 1e-3) {
    estimateSigma(residuals, valid);
  } else {
    dprintf("sigma is stable\n");
  }

  auto valid2 = residuals.size() != valid.size() ? makeValidFlags(valid) : valid;
  assert( valid2.size() == residuals.size() );

  float r_norm = 0.0;
  switch(_loss_func) {
    case LossFunctionType::kHuber:
      r_norm = RunSystemBuilder<HuberOp<float>>(J, residuals, valid2, _sigma, A, b);
      break;
    case LossFunctionType::kTukey:
      r_norm = RunSystemBuilder<TukeyOp<float>>(J, residuals, valid2, _sigma, A, b);
      break;
    case LossFunctionType::kL2:
      r_norm = RunSystemBuilder<L2Op<float>>(J, residuals, valid2, _sigma, A, b);
      break;
    default:
      THROW_ERROR("invalid robust loss");
  }

  return std::sqrt(r_norm);
}

void LinearSystemBuilder::estimateSigma(const ResidualsVector& residuals,
                                        const ValidVector& valid)
{
  getValidResiduals(valid, residuals, _tmp_buffer);
  auto scale = computeRobustStandarDeviation(_tmp_buffer);
  _delta_sigma = std::fabs(scale - _sigma);
  _sigma = scale;
  if(_sigma < 1e-6)
    _sigma = 1e-6; // for the case with zero residuals
}

void LinearSystemBuilder::resetSigma()
{
  _delta_sigma = std::numeric_limits<float>::max();
  _sigma = 1.0f;
}

template <class RobustLoss> static inline
float ComputeWeightedResidualsNorm(const std::vector<float>& residuals,
                                   const std::vector<uint8_t>& valid,
                                   const float& scale)
{
  assert( residuals.size() == valid.size() );
  RobustLoss loss_fn(scale);
  float ret = 0.0f;
  for(size_t i = 0; i < residuals.size(); ++i) {
    float w = loss_fn(residuals[i]) * (float) valid[i];
    ret += w * residuals[i] * residuals[i];
  }

  return ret;
}

float LinearSystemBuilder::run(const ResidualsVector& residuals,
                               const std::vector<uint8_t>& valid)
{
  if(_loss_func != LossFunctionType::kL2 && _delta_sigma > 1e-3) {
    estimateSigma(residuals, valid);
  }

  auto valid2 = residuals.size() != valid.size() ? makeValidFlags(valid) : valid;

  float r_norm = 0.0;
  switch(_loss_func) {
    case LossFunctionType::kHuber:
      r_norm = ComputeWeightedResidualsNorm<HuberOp<float>>(residuals, valid2, _sigma);
      break;
    case LossFunctionType::kTukey:
      r_norm = ComputeWeightedResidualsNorm<TukeyOp<float>>(residuals, valid2, _sigma);
      break;
    case LossFunctionType::kL2:
      r_norm = ComputeWeightedResidualsNorm<L2Op<float>>(residuals, valid2, _sigma);
      break;
    default:
      THROW_ERROR("invalid robust less");
  }

  return std::sqrt(r_norm);
}

template <class L> inline
LinearSystemBuilderReduction<L>::
LinearSystemBuilderReduction(const JacobianVector& J, const ResidualsVector& R,
                             const ValidVector& V, float sigma)
  : _J(J), _residuals(R), _valid(V), _sigma_inv(1.0f/sigma)
  , _H(Hessian::Zero()), _G(Gradient::Zero()), _residuals_norm(0.0f)
{
  assert(_residuals.size() == _valid.size());
}


template <class L> inline
LinearSystemBuilderReduction<L>::
LinearSystemBuilderReduction(LinearSystemBuilderReduction& other, tbb::split)
  : _J(other._J), _residuals(other._residuals), _valid(other._valid)
    , _loss_func(other._loss_func), _sigma_inv(other._sigma_inv)
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

#if DEBUG_THE_STUFF
tbb::mutex MUTEX;
#endif

template <class L> inline
void LinearSystemBuilderReduction<L>::operator()(const tbb::blocked_range<int>& range)
{
  float r_norm = _residuals_norm;

#if WITH_GPL_CODE
  static constexpr int LENGTH = 24;
  alignas(16) float _data[LENGTH];

  memset(_data, 0, sizeof(float)*LENGTH);
#endif

#if DEBUG_THE_STUFF
  MUTEX.lock();
  std::cout << "before\n";
  std::cout << _H << std::endl;
  std::cout << "HHH\n";
#endif

  for(int i = range.begin(); i != range.end(); ++i) {
    if(_valid[i]) {
      float w = _loss_func(_sigma_inv * _residuals[i]);

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

#if DEBUG_THE_STUFF
      std::cout << "pt: " << i << ": " << _J[i] << "\n";
      std::cout << "w: " << w << " r: " << _residuals[i] << std::endl << "\n";
#endif

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

#if DEBUG_THE_STUFF
  std::cout << _H << std::endl;
  std::cout << "num points: " << range.end() - range.begin() << std::endl;
  int dummy;
  std::cin >> dummy;
  MUTEX.unlock();
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

