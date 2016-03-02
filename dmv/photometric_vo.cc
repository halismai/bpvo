#include "dmv/photometric_vo.h"
#include "dmv/patch_util.h"
#include "bpvo/vo_output.h"

#include <SmallVector.h>

#include <opencv2/core/core.hpp>

#if defined(WITH_CERES)
#include "dmv/se3_local_parameterization.h"
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <ceres/loss_function.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/autodiff_cost_function.h>
#endif

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
  void makePointFromDepth(const Mat_<T2,3,3>& K_inv, const T2& z, T2* xyz) const
  {
    Eigen::Map<Vec_<T2,3>> p(xyz);
    p = K_inv * z * formHomog(_uv).template cast<T2>();
  }

  /**
   * create a point from inverse depth
   */
  template <typename T2> inline
  void makePointFromInverseDepth(const Mat_<T2,3,3>& K_inv, const T2& z_i, T2* xyz) const
  {
    return makePointFromDepth(K_inv, T2(1)/z_i, xyz);
  }

  inline const Vec3& xyz() const { return _xyz; }
  inline double depth() const { return _xyz.z(); }

  inline const Vec2& uv() const { return _uv; }

  inline const PatchType& patch() const { return _patch; }
  inline PatchType& patch() { return _patch; }

  inline const T& operator[](int i) const { return _patch[i]; }

  inline void setPatch(const cv::Mat& I, int radius)
  {
    _patch.resize(GetPatchLength(radius));
    int x = (int) _uv.x(), y = (int) _uv.y();
    extractPatch(I.ptr<uint8_t>(), I.step/I.elemSize1(), y, x, radius, _patch.data());
  }

  inline int patchRadius() const
  {
    return static_cast<int>( std::sqrt((float) _patch.size()) / 2.0 );
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

struct PatchErrorWithInverseDepth
{
 public:
  typedef Mat_<double,3,3> Mat33;
  typedef Vec_<double,3>   Vec3;
  typedef Vec_<double,2>   Vec2;

  typedef ceres::Grid2D<double,1> GridType;
  typedef ceres::BiCubicInterpolator<GridType> InterpType;

 public:
  PatchErrorWithInverseDepth(const Mat33& K, const ScenePoint_<double>& scene_point,
                             const InterpType& image)
      : _K(K),  _scene_point(scene_point), _image(image)
        , _radius(_scene_point.patchRadius()) {}

  template <typename T> inline
  bool operator()(const T* const camera, const T* const inv_depth, T* residual) const
  {
    using namespace Eigen;

    Map<const Se3_<T>> pose(camera);

    T u = T( _scene_point.uv().x() );
    T v = T( _scene_point.uv().y() );
    T X[3];
    T z = T(1) / inv_depth[0];
    X[0] = z * ((u - T(_K(0,2))) / T(_K(0,0)));
    X[1] = z * ((v - T(_K(1,2))) / T(_K(1,1)));
    X[2] = z;

    Map<const Vec_<T,3>> X_(X);
    const Vec_<T,2> p = normHomog( _K.cast<T>() * (pose * X_) );

    T i1;
    for(int r = -_radius, i = 0; r <= _radius; ++r)
    {
      T row = p.y() + T(r);
      for(int c = -_radius; c <= _radius; ++c, ++i)
      {
        T col = p.x() + T(c);

        _image.Evaluate(row, col, &i1);
        residual[i] = T(_scene_point[i]) - i1;
      }
    }

    return true;
  }

  static ceres::CostFunction* Create(const Mat33& K, const ScenePoint_<double>& scene_point,
                                     const InterpType& image)
  {
    auto func = new PatchErrorWithInverseDepth(K, scene_point, image);
    int radius = scene_point.patchRadius();
    switch(radius)
    {
      case 1:
        {
          return new ceres::AutoDiffCostFunction<PatchErrorWithInverseDepth, 9, 7, 1>(func);
        } break;

      case 2:
        {
          return new ceres::AutoDiffCostFunction<PatchErrorWithInverseDepth, 25, 7, 1>(func);
        } break;

      default:
        {
          delete func;
          THROW_ERROR(Format("unsupported image patch size %d\n", radius).c_str());
        }
    }
  }

 protected:
  const Mat33& _K;
  const ScenePoint_<double>& _scene_point;
  const InterpType& _image;
  int _radius;
}; // PatchErrorWithInverseDepth

auto PhotometricVo::Impl::addFrame(const VoOutput* vo_output) -> PhotometricVo::Result
{
#if !defined(WITH_CERES)
  THROW_ERROR("compile WITH_CERES");
#endif

  if(_points.empty())
  {
    return init(vo_output);
  }

  const auto image = vo_output->image();
  const auto pose = vo_output->pose();

  ceres::Problem problem;
  Sophus::SE3d se3(pose.cast<double>());
  problem.AddParameterBlock(se3.data(), 7, new Se3LocalParameterization);

  std::vector<double> inv_depth(_points.size());
  for(size_t i = 0; i < _points.size(); ++i)
    inv_depth[i] = 1.0 / _points[i].depth();

  for(size_t i = 0; i < inv_depth.size(); ++i)
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

