#include "dmv/photo_vo.h"
#include "bpvo/imgproc.h"
#include "bpvo/utils.h"

#include <opencv2/core/core.hpp>
#include <vector>
#include <iostream>

#if defined(WITH_CERES)
#include "dmv/patch_error.h"
#include "dmv/se3_local_parameterization.h"
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <ceres/loss_function.h>
#endif // WITH_CERES

namespace bpvo {
namespace dmv {

PhotoVoPatch::PhotoVoPatch(const Mat_<double,3,3>& K, double b, PhotoVoBase::Config config)
    : PhotoVoBase(K, b, config) {}

PhotoVoPatch::~PhotoVoPatch() {}

void PhotoVoPatch::setImageData(const cv::Mat& I_, const cv::Mat& /*D*/)
{
  THROW_ERROR_IF( this->_points.size() == 0, "there are no points" );

  _I0.resize( this->_points.size() );
  const cv::Mat_<uint8_t>& I = (const cv::Mat_<uint8_t>&) I_;
  for(size_t i = 0; i < _I0.size(); ++i)
  {
    int y = this->_points[i].uv().y();
    int x = this->_points[i].uv().x();

    for(int r = -1, j = 0; r <= 1; ++r)
    {
      for(int c = -1; c <= 1; ++c, ++j)
      {
        _I0[i][j] = PixelScale * I(y+r,x+c);
      }
    }
  }
}

PhotoVoBase::Result
PhotoVoPatch::estimatePose(const cv::Mat& I, const Mat_<double,4,4>& T_init)
{
#if !defined(WITH_CERES)
  THROW_ERROR("compile WITH_CERES\n");
#endif

  THROW_ERROR_IF( I.channels() > 1 || I.type() != CV_8U, "image must be uint8_t grayscale");

  typename PatchError::GridType grid(I.ptr<uint8_t>(), 0, I.rows, 0, I.cols);
  typename PatchError::InterpType interp(grid);

  ceres::Problem problem;
  Sophus::SE3d se3(T_init.cast<double>());
  problem.AddParameterBlock(se3.data(), 7, new Se3LocalParameterization);

  bool use_robust = true;
  ceres::LossFunction* loss = use_robust ? new ceres::SoftLOneLoss(10.0 * PixelScale) : NULL;

  for(size_t i = 0; i < _I0.size(); ++i)
    problem.AddResidualBlock(PatchError::Create(_K, this->_points[i].xyz(), interp, _I0[i].data()), loss, se3.data());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.function_tolerance = _config.functionTolerance;
  options.gradient_tolerance = _config.parameterTolerance;
  options.parameter_tolerance = _config.gradientTolerance;
  options.max_num_iterations = _config.maxIterations;
  options.use_nonmonotonic_steps = true;
  options.num_threads = 4;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << std::endl;
  //std::cout << summary.BriefReport() << std::endl;

  return Result(se3.matrix());
}

} // dmv
} // bpvo
