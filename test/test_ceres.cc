#include <bpvo/debug.h>

#if defined(WITH_CERES)

#include <bpvo/math_utils.h>
#include <bpvo/utils.h>
#include <bpvo/types.h>

#include <ceres/sized_cost_function.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/autodiff_local_parameterization.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <ceres/rotation.h>

#include <dmv/reprojection_error.h>
#include <dmv/se3_ceres.h>

#include <iostream>

using namespace bpvo;

template <typename T> using Mat3_ = Eigen::Matrix<T,3,3>;
template <typename T> using Mat4_ = Eigen::Matrix<T,4,4>;
template <typename T, int N> using Vec_ = Eigen::Matrix<T,N,1>;


class ReprojectionErrorFunctor
{
 public:
  typedef Mat3_<double> Mat3;
  typedef Mat4_<double> Mat4;
  typedef Vec_<double,2> Point2;
  typedef Vec_<double,3> Point3;
  typedef Vec_<double,6> ParameterVector;

  static constexpr int ResidualSize = 2;
  static constexpr int ParamSize = 6;

 public:
  ReprojectionErrorFunctor(const double* K, const double* X, const double* x)
      : _K(K), _X(X), _x(x) {}

  template <typename T>
  bool operator()(const T* params, T* residuals) const
  {
    T X[3];
    T X0[3] = { T(_X[0]), T(_X[1]), T(_X[2]) };
    ceres::AngleAxisRotatePoint(params, X0, X);
    X[0] += params[3];
    X[1] += params[4];
    X[2] += params[5];

    const Eigen::Map<const Mat3_<double>, Eigen::Aligned> K(_K);

    const T fx = T( K(0,0) );
    const T fy = T( K(1,1) );
    const T cx = T( K(0,2) );
    const T cy = T( K(1,2) );

    residuals[0] = _x[0] - ((fx*X[0])/X[2] + cx);
    residuals[1] = _x[1] - ((fy*X[1])/X[2] + cy);

    return true;
  }

 protected:
  const double* _K;
  const double* _X;
  const double* _x;
}; // ReprojectionErrorFunctor

typedef typename EigenAlignedContainer<Vec_<double,2>>::type Point2Vector;
typedef typename EigenAlignedContainer<Vec_<double,3>>::type Point3Vector;

void makePoints(int N, const Mat3_<double>& K, Point3Vector& X, Point2Vector& x)
{
  Mat3_<double> K_inv = K.inverse();

  X.resize(N);
  x.resize(N);

  for(int i = 0; i < N; ++i)
  {
    x[i][0] = randrange<double>(0, 640-1);
    x[i][1] = randrange<double>(0, 480-1);

    double z = randrange<double>(0.5, 25.0);
    X[i] = K_inv * z * Vec_<double,3>(x[i].x(), x[i].y(), 1.0);
  }
}

Point2Vector projectPoints(const Mat3_<double>& K, const Mat4_<double>& T,
                           const Point3Vector& X)
{
  const Eigen::Matrix<double,3,4> P = K * T.block<3,4>(0,0);
  Point2Vector ret(X.size());

  for(size_t i = 0; i < X.size(); ++i)
  {
    Vec_<double,3> Y = P * Vec_<double,4>(X[i].x(), X[i].y(), X[i].z(), 1.0);
    ret[i] = Y.head<2>() / Y[2];
  }

  return ret;
}

template <class ObjectiveFunctor> inline
Vec_<double,ObjectiveFunctor::ParamSize>
Run(const Mat3_<double>& K, const Point2Vector& x, const Point3Vector& X, bool local_param = false)
{
  Vec_<double,ObjectiveFunctor::ParamSize> p; p.setZero();

  if(local_param) {
    p[0] = 1.0;
  }

  ceres::Problem problem;

  if(local_param) {
    //ceres::LocalParameterization* auto_diff = new ceres::AutoDiffLocalParameterization
    //    <dmv::Se3LocalParameterization, dmv::Se3LocalParameterization::NUM_PARAMS,
    //    dmv::Se3LocalParameterization::DOF>();

    problem.AddParameterBlock(p.data(), ObjectiveFunctor::ParamSize,
                              new dmv::Se3LocalParameterization);
  }

  for(size_t i = 0; i < x.size(); ++i)
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<
                             ObjectiveFunctor,ObjectiveFunctor::ResidualSize,ObjectiveFunctor::ParamSize>
                             (new ObjectiveFunctor(K.data(), X[i].data(), x[i].data())), NULL, p.data());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << std::endl;

  return p;
}


int main()
{
  Point3Vector X;
  Point2Vector x;

  Mat3_<double> K; K << 615.0, 0, 320, 0, 615.0, 240, 0, 0, 1;
  makePoints(0.5*640*480, K, X, x);

  Mat4_<double> T(Mat4_<double>::Identity());
  T(0,3) = 0.05;
  T(1,3) = 0.05;
  T(2,3) = 0.05;

  T.block<3,3>(0,0) = math::EulerVectorToRotationMatrix(1, 2, 3);

  auto y = projectPoints(K, T, X);
  auto p = Run<ReprojectionErrorFunctor>(K, y, X);

  std::cout << "p: " << p.transpose() << std::endl;

  auto p2 = Run<dmv::ReprojectionErrorSe3>(K, y, X, true);
  std::cout << "p" << p2.transpose() << std::endl;

  return 0;
}

#else
int main()
{
  Fatal("compile WITH_CERES");
}
#endif
