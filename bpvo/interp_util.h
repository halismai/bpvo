#ifndef INTERP_UTIL_H
#define INTERP_UTIL_H

#include <bpvo/types.h>

namespace bpvo {

template <typename T = float>
class BilinearInterp
{
 public:
  typedef Eigen::Matrix<T,4,1> Vector4;

 public:
  /**
   */
  BilinearInterp() : _stride(0) {}

  /**
   * \param warp a functor to warp the points (see bpvo/warps.h)
   * \param points a vector of 3D points
   * \param rows, cols the size of the image
   */
  template <class Warp, class PointVector> inline
  void init(const Warp& warp, const PointVector& points, int rows, int cols)
  {
    _stride = cols;
    resize(points.size());

    for(size_t i = 0; i < points.size(); ++i) {
      auto p = warp(points[i]);
      float xf = p.x(), yf = p.y();
      int xi = static_cast<int>(xf), yi = static_cast<int>(yf);
      xf -= xi;
      yf -= yi;
      _valid[i] = xi>=0 && xi<cols-1 && yi>=0 && yi<rows-1;
      _inds[i] = yi*cols + xi;
      _interp_coeffs[i] = Vector4((1.0-yf)*(1.0-xf),
                                  (1.0-yf)*xf,
                                  yf*(1.0-xf),
                                  yf*xf);
    }
  }

  /**
   * \return interpolated value at point 'i'
   */
  template <class ImageT> inline
  T operator()(const ImageT* ptr, int i) const
  {
    return _valid[i] ? _interp_coeffs[i].dot(load_data(ptr, i)) : T(0);
  }

  inline const std::vector<uint8_t>& valid() const { return _valid; }
  inline       std::vector<uint8_t>& valid()      { return _valid; }

 protected:
  // [(1-yf)*(1-xf), (1-yf)*xf, yf*(1-xf), xf]
  typename EigenAlignedContainer<Vector4>::type _interp_coeffs;
  std::vector<int> _inds;       //< yi*cols + xi
  std::vector<uint8_t>  _valid; //< valid flags
  int _stride;

  void resize(size_t n)
  {
    _interp_coeffs.resize(n);
    _inds.resize(n);
    _valid.resize(n);
  }

  template <class ImageT> inline
  Vector4 load_data(const ImageT* ptr, int i) const
  {
    auto p = ptr + _inds[i];
    return Vector4(*p, *(p + 1), *(p + _stride), *(p + _stride + 1));
  }

}; // BilinearInterp


}; // bpvo


#endif // INTERP_UTIL_H
