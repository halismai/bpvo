#ifndef BPVO_DMV_PHOTO_WORLD_POINT_H
#define BPVO_DMV_PHOTO_WORLD_POINT_H

#include <bpvo/eigen.h>

namespace bpvo {
namespace dmv {

class WorldPoint
{
 public:
  /**
   */
  WorldPoint() {}

  /**
   */
  WorldPoint(const Mat_<double,3,3>& K_inv, const Vec_<double,2>& uv, double z)
      : _uv(uv), _xyz(K_inv * z * formHomog(uv)) {}

  /**
   * create a 3D point using the depth 'z' and inverse camera matrix K_inv
   */
  template <typename T> inline
  void makePoint(const Mat_<T,3,3>& K_inv, const T& z, T* xyz_) const
  {
    Eigen::Map<Vec_<T,3>> p(xyz_);
    p = z * K_inv * _uv.cast<T>();
  }

  inline const Vec_<double,2>& uv() const { return _uv; }
  inline const Vec_<double,3>& xyz() const { return _xyz; }

 protected:
  Vec_<double,2> _uv;
  Vec_<double,3> _xyz;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // WorldPoint

} // dmv
} // bpvo

#endif // BPVO_DMV_PHOTO_WORLD_POINT_H
