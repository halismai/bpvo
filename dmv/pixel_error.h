#ifndef BPVO_DMV_PIXEL_ERROR_H
#define BPVO_DMV_PIXEL_ERROR_H
#if defined(WITH_CERES)

#include <ceres/cubic_interpolation.h>
#include <dmv/se3_local_parameterization.h>
#include <bpvo/eigen.h>

namespace ceres {
class CostFunction;
}; // ceres

namespace bpvo {
namespace dmv {

/**
 * a single pixel error
 */
class PixelError
{
 public:
  static constexpr double PixelScale = 1.0 / 255.0;
  static constexpr int ResidualSize = 1;
  static constexpr int NumParameters = Se3_<double>::num_parameters;

  typedef ceres::Grid2D<uint8_t, 1> GridType;
  typedef ceres::BiCubicInterpolator<GridType> InterpType;

 public:
  PixelError(const Mat_<double,3,3>& K, const Vec_<double,3>& X,
             const InterpType& image, double i0)
      : _K(K), _X(X), _image(image), _i0(i0) {}


  template <typename T> inline
  bool operator()(const T* params, T* residual) const
  {
    using namespace Eigen;

    Map<const Se3_<T>> pose(params);
    const Vec_<T,2> x = normHomog(_K.cast<T>() * (pose * _X.cast<T>()));

    T i1;
    _image.Evaluate(x[1], x[0], &i1);
    residual[0] = PixelScale*i1 - T(_i0);

    return true;
  }

  static ceres::CostFunction* Create(const Mat_<double,3,3>& K, const Vec_<double,3>& X,
                                     const InterpType& image, double i0);

 protected:
  const Mat_<double,3,3>& _K;
  const Vec_<double,3>& _X;
  const InterpType& _image; //< the image to sample from
  double _i0; //< reference frame intensity

}; // PixelError

}; // dmv
}; // bpvo

#endif // WITH_CERES
#endif // BPVO_DMV_PIXEL_ERROR_H
