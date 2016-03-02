#include "dmv/photo_vo.h"

#include "bpvo/imgproc.h"
#include "bpvo/utils.h"

#include <opencv2/core/core.hpp>
#include <vector>
#include <iostream>

#if defined(WITH_CERES)
#include "dmv/pixel_error.h"
#include "dmv/se3_local_parameterization.h"

#include <ceres/problem.h>
#include <ceres/solver.h>
#include <ceres/loss_function.h>


#endif // WITH_CERES

namespace bpvo {
namespace dmv {

template <typename T>
struct IsLocalMax_
{
  inline IsLocalMax_() {}

  inline IsLocalMax_(const T* ptr, int stride, int radius)
      : _ptr(ptr), _stride(stride), _radius(radius)
  {
    if(_radius > 0) assert( _stride > 0);
  }

  inline void setStride(int s) { _stride = s; }
  inline void setRadius(int r) { _radius = r; }
  inline void setPointer(const T* p) { _ptr = p; }

  inline bool operator()(int row, int col) const
  {
    if(_radius <= 0)
      return true;

    switch(_radius)
    {
      case 1: // 3x3
        {
          const T* p0 = _ptr + row*_stride + col;
          const T* p1 = p0 - _stride;
          const T* p2 = p0 + _stride;
          auto v = *p0;

          return
              (v > p0[-1]) &&                (v > p0[1]) &&
              (v > p1[-1]) && (v > p1[0]) && (v > p1[1]) &&
              (v > p2[-1]) && (v > p2[0]) && (v > p2[1]);
        } break;

        // TODO case 2
      default:
        {
          // generic implementation for any radius
          auto v = *(_ptr + row*_stride + col);
          for(int r = -_radius; r <= _radius; ++r)
            for(int c = -_radius; c <= _radius; ++c)
              if(!(!r && !c) && *(_ptr + (row+r)*_stride + col + c) >= v)
                return false;

          return true;
        }
    }
  }

 private:
  const T* _ptr;
  int _stride, _radius;
}; // IsLocalMax_

PhotoVo::PhotoVo(const Eigen::Matrix<double,3,3>& K, double b, Config conf)
    : _K(K), _baseline(b), _config(conf) {}

void PhotoVo::setTemplate(const cv::Mat& I, const cv::Mat& D)
{
  THROW_ERROR_IF( I.type() != cv::DataType<uint8_t>::type || I.channels() > 1,
                 "image must CV_8UC1" );

  THROW_ERROR_IF( D.type() != cv::DataType<float>::type, "disparity must be CV_32F" );

  _I0.clear();
  _X0.clear();

  cv::Mat G(I.size(), CV_16S);
  for(int y = 1; y < I.rows - 2; ++y)
  {
    auto srow0 = I.ptr<uint8_t>(y-1),
         srow1 = I.ptr<uint8_t>(y+1),
         srow = I.ptr<uint8_t>(y);
    auto drow = G.ptr<short>(y);
    for(int x = 1; x < I.cols - 2; ++x)
    {
      drow[x] = std::abs(srow[x+1] - srow[x-1]) + std::abs(srow1[x] - srow0[x]);
    }
  }

  const IsLocalMax_<short> is_local_max(G.ptr<short>(), G.step/G.elemSize1(), _config.nonMaxSuppRadius);
  double fx = _K(0,0),
       fy = _K(1,1),
       cx = _K(0,2),
       cy = _K(1,2),
       bf = static_cast<double>( _baseline * fx );

  for(int y = 1; y < I.rows - 2; ++y)
    for(int x = 1; x < I.cols - 2; ++x)
    {
      if(G.at<short>(y,x) > _config.minSaliency && is_local_max(y,x))
      {
        if(D.at<float>(y,x) > 0.0f)
        {
          double Z = bf / D.at<float>(y,x),
                 X = (x - cx) * Z / fx,
                 Y = (y - cy) * Z / fy;

          _X0.push_back( Eigen::Matrix<double,3,1>(X, Y, Z) );
          _I0.push_back( PixelError::PixelScale * static_cast<double>(I.at<uint8_t>(y,x)));
        }
      }
    }
}


struct ImageGradient
{
  ImageGradient(const cv::Mat& I) { compute(I); }

  void compute(const cv::Mat& I)
  {
    THROW_ERROR_IF( I.type() != cv::DataType<uint8_t>::type, "image must uint8_t" );

    imgradient(I, Ix, ImageGradientDirection::X);
    imgradient(I, Iy, ImageGradientDirection::Y);
  }

  cv::Mat_<float> Ix;
  cv::Mat_<float> Iy;
}; // ImageGradient

auto PhotoVo::estimatePose(const cv::Mat& I, const Matrix44& T_init) const -> Result
{
#if !defined(WITH_CERES)
  THROW_ERROR("compile WITH_CERES\n");
#endif

  THROW_ERROR_IF( I.channels() > 1 || I.type() != CV_8U, "image must be uint8_t grayscale");

  typename PixelError::GridType grid(I.ptr<uint8_t>(), 0, I.rows, 0, I.cols);
  typename PixelError::InterpType interp(grid);

  ceres::Problem problem;
  Sophus::SE3d se3(T_init.cast<double>());
  problem.AddParameterBlock(se3.data(), 7, new Se3LocalParameterization);

  for(size_t i = 0; i < _I0.size(); ++i)
    problem.AddResidualBlock(PixelError::Create(_K, _X0[i], interp, _I0[i]),
                             _config.withRobust ? new ceres::HuberLoss(10.0/255.0) : NULL, se3.data());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.function_tolerance = 1e-6;
  options.gradient_tolerance = 1e-6;
  options.parameter_tolerance = 1e-6;
  options.max_num_iterations = _config.maxIterations;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  //std::cout << summary.FullReport() << std::endl;
  std::cout << summary.BriefReport() << std::endl;

  Result ret;
  ret.pose = se3.matrix().cast<float>();

  return ret;
}

} // dmv
} // bpvo

