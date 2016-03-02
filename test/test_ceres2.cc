#include <bpvo/debug.h>
#if !defined(WITH_CERES)
int main() { Fatal("compile WITH_CERES\n"); }
#else

#include <ceres/local_parameterization.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <sophus/se3.hpp>

#include <bpvo/types.h>
#include <bpvo/utils.h>
#include <bpvo/math_utils.h>

#include <dmv/se3_local_parameterization.h>

#include <iostream>

template <typename T, int M, int N> using Mat_ = Eigen::Matrix<T,M,N>;
template <typename T, int M> using Vec_ = Eigen::Matrix<T,M,1>;

template <typename T> using Se3_ = Sophus::SE3Group<T>;

typedef typename bpvo::EigenAlignedContainer<Vec_<double,3>>::type Point3Vector;
typedef typename bpvo::EigenAlignedContainer<Vec_<double,2>>::type Point2Vector;

template <class Derived> static inline
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime - 1, 1>
normHomog(const Eigen::MatrixBase<Derived>& p)
{
  static_assert( Derived::RowsAtCompileTime != Eigen::Dynamic,
                "matrix size must be known at compile time" );

  constexpr int R = Derived::RowsAtCompileTime;
  typedef typename Derived::Scalar T;

  return Eigen::Matrix<T, R-1, 1>( (T(1) / p[R-1]) * p.template head<R-1>());
}


/**
 * Local parameter for SE3
 */
class Se3LocalParameterization : public ceres::LocalParameterization
{
  typedef Sophus::SE3d SE3;

  typedef Vec_<double,6> Vec6;
  typedef Mat_<double,6,7> Mat67;

 public:
  virtual ~Se3LocalParameterization() {}

  /**
   * \param T pointer to the pose
   * \param delta_T_ the SE3 local parameterization 6 numbers
   * \param T_ret the composed pose 4x4
   */
  virtual bool Plus(const double* T, const double* delta, double* T_ret_) const
  {
    using namespace Eigen;

    Map<SE3> T_ret(T_ret_);
    T_ret = Map<const SE3>(T) * SE3::exp(Map<const Vec6>(delta));

    return true;
  }

  virtual bool ComputeJacobian(const double* T_, double* J_) const
  {
    using namespace Eigen;

    Map<Mat67> J(J_);
    J = Map<const SE3>(T_).internalJacobian().transpose();

    return true;
  }

  virtual int GlobalSize() const { return SE3::num_parameters; /* 7 */ }

  virtual int LocalSize() const { return SE3::DoF; /* 6 */ }
};


/**
 * Image re-reprojection error with the pose represented with SE3
 */
class ReprojErrorSe3
{

  typedef ReprojErrorSe3 Self;

 public:
  typedef Mat_<double,3,3> Mat3;

 public:
  /**
   */
  ReprojErrorSe3(const Mat3& K, const Vec_<double,3>& X, const Vec_<double,2>& x)
      : _K(K), _X(X), _x(x) {}

  template <typename T> inline
  bool operator()(const T* params, T* residual) const
  {
    using namespace Eigen;

    Map<const Se3_<T>> pose(params);
    Map<Vec_<T,2>> r(residual);
    r = normHomog(_K.cast<T>() * (pose * _X.cast<T>())) - _x.cast<T>();
    return true;
  }

  static ceres::CostFunction*
  Create(const Mat3& K, const Vec_<double,3>& X, const Vec_<double,2>& x)
  {
    return new ceres::AutoDiffCostFunction<Self, 2, Sophus::SE3d::num_parameters>(
        new Self(K, X, x));
  }

 private:
  const Mat3& _K;
  const Vec_<double,3>& _X;
  const Vec_<double,2>& _x;
}; // ReprojErrorSe3

inline Point3Vector makePoints(int N, const Mat_<double,3,3>& K)
{
  const Mat_<double,3,3> K_inv = K.inverse();

  auto rand_image_point = [=]()
  {
    return Vec_<double,3>(
        bpvo::randrange<double>(0,640-1), bpvo::randrange<double>(0,480-1), 1);
  };

  Point3Vector X(N);
  for(int i = 0; i < N; ++i)
    X[i] = K_inv * bpvo::randrange<double>(0.5,25.0) * rand_image_point();

  return X;
}

inline Point2Vector
projectPoints(const Mat_<double,3,3>& K, const Se3_<double>& pose, const Point3Vector& X)
{
  Point2Vector x(X.size());
  for(size_t i = 0; i < x.size(); ++i)
    x[i] = normHomog( K * (pose * X[i]) );

  return x;
}

int main()
{
  int N = 100*100;
  Mat_<double,3,3> K; K << 615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0;

  Mat_<double,3,3> R = bpvo::math::EulerVectorToRotationMatrix<double>(1,2,3);
  Vec_<double,3> t = Vec_<double,3>(0.01, 0.02, 0.03);
  Se3_<double> T_true(R, t);
  Se3_<double> T_est;
  auto X = makePoints(N, K);
  auto x = projectPoints(K, T_true, X);


  ceres::Problem problem;
  problem.AddParameterBlock(T_est.data(), 7, new bpvo::dmv::Se3LocalParameterization_<double>);
  for(size_t i = 0; i < X.size(); ++i)
    problem.AddResidualBlock(ReprojErrorSe3::Create(K, X[i], x[i]), NULL, T_est.data());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << std::endl;

  Mat_<double,4,4> T_err = (Mat_<double,4,4>::Identity() -
                            (T_est.matrix().inverse() * T_true.matrix()));
  std::cout << "ERROR: " << T_err.array().abs().maxCoeff() << std::endl;

  return 0;
}

#endif
