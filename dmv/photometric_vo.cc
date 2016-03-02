#include "dmv/photometric_vo.h"
#include "dmv/patch_util.h"

#include "bpvo/vo_output.h"

#include <SmallVector.h>

#include <opencv2/core/core.hpp>

namespace bpvo {
namespace dmv {

template <typename T>
struct ScenePoint_
{
  typedef Vec_<T,2> Vec2;
  typedef Vec_<T,3> Vec3;
  typedef Mat_<T,3,3> Mat33;

  typedef llvm::SmallVector<T,16> PatchType;

 public:
  /**
   */
  ScenePoint_() {}

  /**
   * Create a point from the inverse camera matrix, image projection and depth
   */
  ScenePoint_(const Mat33& K_inv, const Vec2& uv, T depth)
      : _uv(uv), _xyz(K_inv * depth * formHomog(uv)) {}

  /**
   * Create point and assign patch
   */
  ScenePoint_(const Mat33& K_inv, const Vec2& uv, T depth, const PatchType& p)
      : ScenePoint_(K_inv, uv, depth)
  {
    _patch = p;
  }

  template <typename T2> inline
  void makePointFromDepthh(const Mat_<T2,3,3>& K_inv, const T2& z, T2* xyz) const
  {
    Eigen::Map<Vec_<T,3>> p(xyz);
    p = K_inv * z * _uv.template cast<T2>();
  }

  /**
   * create a point from inverse depth
   */
  template <typename T2> inline
  void makePointFromInverseDepth(const Mat_<T2,3,3>& K_inv, const T2& z_i, T2* xyz) const
  {
    return makePointFromDepthh(K_inv, T2(1)/z_i, xyz);
  }

  inline const Vec3& xyz() const { return _xyz; }

  inline const PatchType& patch() const { return _patch; }
  inline PatchType& patch() { return _patch; }

  inline const T& operator[](int i) const { return _patch[i]; }

  inline void setPatch(const cv::Mat& I, int radius)
  {
    _patch.resize(GetPatchLength(radius));
    int x = (int) _uv.x(), y = (int) _uv.y();
    extractPatch(I.ptr<uint8_t>(), I.step/I.elemSize1(), y, x, radius, _patch.data());
  }

 protected:
  Vec2 _uv;         // uv location in the image
  Vec3 _xyz;        // position in 3D
  PatchType _patch; // image patch

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // WorldPoint_



struct PhotometricVo::Impl
{
  Impl(const Mat33&, double b, PhotometricVoConfig);
  PhotometricVo::Result addFrame(const VoOutput*);
  PhotometricVo::Result init(const VoOutput*);

  Mat33 _K, _K_inv;
  double _b;
  PhotometricVoConfig _config;

  typename EigenAlignedContainer<ScenePoint_<double>>::type _points;
}; // Impl

PhotometricVo::PhotometricVo(const Mat33& K, double b, PhotometricVoConfig config)
  : _impl(make_unique<Impl>(K, b, config)) {}

PhotometricVo::~PhotometricVo() {}

auto PhotometricVo::addFrame(const VoOutput* vo_output) -> Result
{
  return _impl->addFrame(vo_output);
}

PhotometricVo::Impl::Impl(const Mat33& K, double b, PhotometricVoConfig config)
  : _K(K), _K_inv(K.inverse()), _b(b), _config(config) {}


auto PhotometricVo::Impl::addFrame(const VoOutput* vo_output) -> PhotometricVo::Result
{
  if(_points.empty())
  {
    return init(vo_output);
  }
  else
  {
  }
}

auto PhotometricVo::Impl::init(const VoOutput* vo_output) -> PhotometricVo::Result
{
  const auto& pc = vo_output->pointCloud();
  _points.resize(0);
  _points.reserve(pc.size());

  const auto& image = vo_output->image();

  for(const auto& pt : pc)
  {
    if(pt.weight() > _config.minWeight)
    {
      Vec_<double,2> uv = normHomog( _K * pt.xyzw().cast<double>().head<3>() );
      double z = pt.xyzw()[2];
      _points.push_back(ScenePoint_<double>(_K_inv, uv, z));
      _points.back().setPatch(image, _config.patchRadius);
    }
  }

  Result ret;
  ret.pose.setIdentity();
  ret.points.resize(_points.size());
  for(size_t i = 0; i < _points.size(); ++i)
    ret.points[i] = _points[i].xyz();

  return ret;
}

} // dmv
} // bpvo
