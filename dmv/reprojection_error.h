#ifndef BPVO_DMV_REPROJECTION_ERROR_H
#define BPVO_DMV_REPROJECTION_ERROR_H
#if defined(WITH_CERES)

namespace ceres {
class CostFunction;
}; // ceres

namespace bpvo {
namespace dmv {

class ReprojectionErrorSe3
{
 public:
  static constexpr int ResidualSize = 2;
  static constexpr int ParamSize = 7;

 public:
  ReprojectionErrorSe3(const double* K_matrix, const double* x3d,
                       const double* x_observation);

  template <class T>
  bool operator()(const T* const params, T* residuals) const;

  static ceres::CostFunction* Create(const double* K, const double* X,
                                     const double* x);

 protected:
  const double* _K;
  const double* _X;
  const double* _x;
}; //

}; // dmv
}; // bpvo

#endif // WITH_CERES
#endif
