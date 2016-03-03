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

  inline void setPatch(const cv::Mat& I, int radius, double pixel_scale)
  {
    _patch.resize(GetPatchLength(radius));

    int x = (int) _uv.x(), y = (int) _uv.y();
    size_t stride = I.step / I.elemSize1();
    extractPatch(I.ptr<uint8_t>(), stride, I.rows, I.cols, y, x, radius,
                 _patch.data(), pixel_scale);
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

  typedef ceres::Grid2D<uint8_t,1> GridType;
  typedef ceres::BiCubicInterpolator<GridType> InterpType;

 public:
  PatchErrorWithInverseDepth(const Mat33& K, const ScenePoint_<double>& scene_point,
                             const InterpType& image, double pixel_scale,
                             bool with_spatial_weighting)
      : _K(K),  _scene_point(scene_point), _image(image)
      , _radius(_scene_point.patchRadius()), _pixel_scale(pixel_scale)
      , _with_spatial_weighting(with_spatial_weighting)
  {
    if(_with_spatial_weighting) {
      double s_x = 1.0;
      double s_y = 1.0;
      double a = 1.0;

      double sum = 0.0;
      for(int r = -_radius; r <= _radius; ++r)
      {
        double d_r = (r*r) / s_x;

        for(int c = -_radius; c <= _radius; ++c)
        {
          double d_c = (c*c) / s_y;
          double w = a * std::exp( -0.5*(d_r + d_c) );
          sum += w;
          _weight_table.push_back(w);
        }
      }

      for(size_t i = 0; i < _weight_table.size(); ++i)
        _weight_table[i] = _weight_table[i] / sum;
    }

  }

  template <typename T> inline
  bool operator()(const T* const camera, const T* const inv_depth, T* residual) const
  {
    using namespace Eigen;

    Map<const Se3_<T>> pose(camera);

    T u = T( _scene_point.uv().x() );
    T v = T( _scene_point.uv().y() );
    T z = T(1) / inv_depth[0];

    T X[3];
    X[0] = (z * ((u - T(_K(0,2)))) / T(_K(0,0)));
    X[1] = (z * ((v - T(_K(1,2)))) / T(_K(1,1)));
    X[2] = z;

    Map<const Vec_<T,3>> X_(X);
    const Vec_<T,2> p = normHomog( _K.cast<T>() * (pose * X_) );

    std::cout << "val: " << _scene_point.uv().transpose() << std::endl;
    std::cout << "p0:" << p[0] << std::endl;
    std::cout << "p1:" << p[1] << std::endl;


    T i1;
    for(int r = -_radius, i = 0; r <= _radius; ++r)
    {
      T row = p.y() + T(r);
      for(int c = -_radius; c <= _radius; ++c, ++i)
      {
        T col = p.x() + T(c);

        _image.Evaluate(row, col, &i1);
        residual[i] = T(_scene_point[i]) - T(_pixel_scale)*i1;
        //residual[i] = residual[i] / T(9);

        if(r == 0 && c == 0 ) {
          std::cout << "r: " << row << "\n";
          std::cout << "c: " << col << "\n";
          std::cout << "residual: " << residual[i] << std::endl;
          std::cout << "i1: " << i1 << std::endl;
          std::cout << "scene point: " << _scene_point[i] << "\n" << i1 << "\n\n";
        }

        if(_with_spatial_weighting) {
          residual[i] = residual[i] * _weight_table[i];
        }
      }
    }

    throw "bye";

    return true;
  }

  static ceres::CostFunction* Create(const Mat33& K, const ScenePoint_<double>& scene_point,
                                     const InterpType& image, double pixel_scale, bool with_spatial_weighting)
  {
    auto func = new PatchErrorWithInverseDepth(K, scene_point, image,
                                               pixel_scale, with_spatial_weighting);
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
  double _pixel_scale;
  bool _with_spatial_weighting;

  llvm::SmallVector<double, 16> _weight_table;
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
  Sophus::SE3d se3(/*Mat_<double,4,4>::Identity()*/ pose.cast<double>());
  problem.AddParameterBlock(se3.data(), 7, new Se3LocalParameterization);

  typename PatchErrorWithInverseDepth::GridType grid(
      image.ptr<uint8_t>(), 0, image.rows, 0, image.cols);
  typename PatchErrorWithInverseDepth::InterpType interp(grid);

  std::vector<double> inv_depth(_points.size());
  for(size_t i = 0; i < _points.size(); ++i)
    inv_depth[i] = 1.0 / _points[i].depth();

  Result ret;
  ret.pointsRaw.resize(_points.size());
  for(size_t i = 0; i < _points.size(); ++i)
    ret.pointsRaw[i] = _points[i].xyz();

  bool use_robust = true;
  double robust_param = 10.0 * _config.intensityScale;

  printf("number of points %zu\n", _points.size());
  for(size_t i = 0; i < inv_depth.size(); ++i)
  {
    ceres::CostFunction* cost = PatchErrorWithInverseDepth::Create(
        _K, _points[i], interp, _config.intensityScale, _config.withSpatialWeighting);
    ceres::LossFunction* loss = use_robust ? new ceres::SoftLOneLoss(robust_param) : NULL;
    problem.AddResidualBlock(cost, loss, se3.data(), &inv_depth[i]);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.function_tolerance = _config.functionTolerance;
  options.gradient_tolerance = _config.parameterTolerance;
  options.parameter_tolerance = 1e-6;
  options.max_num_iterations = _config.maxIterations;
  options.num_threads = 1;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << std::endl;

  ret.pose = se3.matrix();
  ret.points.resize(inv_depth.size());
  for(size_t i = 0; i < _points.size(); ++i)
    _points[i].makePointFromInverseDepth(_K_inv, inv_depth[i], ret.points[i].data());

  return ret;
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

      _points.back().setPatch(image, _config.patchRadius, _config.intensityScale);
    }
  }


  Result ret;
  ret.pose.setIdentity();
  ret.points.resize(_points.size());
  ret.pointsRaw.resize(_points.size());
  for(size_t i = 0; i < _points.size(); ++i) {
    ret.points[i] = _points[i].xyz();
    ret.pointsRaw[i] = ret.points[i];
  }

  return ret;
}

} // dmv
} // bpvo

