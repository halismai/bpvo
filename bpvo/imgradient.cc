#include "bpvo/imgproc.h"
#include "bpvo/utils.h"

#include <cstring>

#include <Eigen/Core>

namespace bpvo {

template <typename TSrc, typename TDst> static inline
void xgradient(const TSrc* src, int rows, int cols, TDst* dst)
{
  assert_is_floating_point<TDst>();

  using namespace Eigen;
  typedef Map< Matrix<TDst, Dynamic, Dynamic, RowMajor>, Eigen::Aligned> DstMap;
  typedef Map< const Matrix<TSrc, Dynamic, Dynamic, RowMajor>, Eigen::Aligned> SrcMap;

  DstMap Ix(dst, rows, cols);
  const SrcMap I(src, rows, cols);

  Ix.col(0) = TDst(0.5) * (I.col(1).template cast<TDst>() - I.col(0).template cast<TDst>());
  Ix.block(0, 1, rows, cols - 2) =
      TDst(0.5) * (I.block(0, 2, rows, cols - 2).template cast<TDst>() -
                 I.block(0, 0, rows, cols - 2).template cast<TDst>());
  Ix.col(cols-1) = TDst(0.5) * (I.col(cols-1).template cast<TDst>() -
                               I.col(cols-2).template cast<TDst>());
}

template <typename TSrc, typename TDst> static inline
void ygradient(const TSrc* src, int rows, int cols, TDst* dst)
{
  assert_is_floating_point<TDst>();

  using namespace Eigen;
  typedef Map<Matrix<TDst, Dynamic, Dynamic, RowMajor>, Aligned> DstMap;
  typedef Map< const Matrix<TSrc, Dynamic, Dynamic, RowMajor>, Aligned> SrcMap;

  DstMap Iy(dst, rows, cols);
  const SrcMap I(src, rows, cols);

  Iy.row(0) = TDst(0.5) * (I.row(1).template cast<TDst>() - I.row(0).template cast<TDst>());

  Iy.block(1, 0, rows - 2, cols) =
      TDst(0.5) * (I.block(2, 0, rows - 2, cols).template cast<TDst>() -
                 I.block(0, 0, rows - 2, cols).template cast<TDst>());

  Iy.row(rows - 1) = TDst(0.5) * (I.row(rows - 1).template cast<TDst>() -
                                  I.row(rows - 2).template cast<TDst>());
}


void imgradient(const cv::Mat& src, cv::Mat& dst, ImageGradientDirection dir)
{
  THROW_ERROR_IF( src.channels() > 1, "imgradient works with single channel only" );

  typedef float DType;
  dst.create(src.size(), CV_MAKETYPE(cv::DataType<DType>::depth, 1));

  auto src_ptr = src.ptr<const uint8_t>();
  auto dst_ptr = dst.ptr<DType>();
  auto rows = src.rows, cols = src.cols;

  if(dir == ImageGradientDirection::X)
    xgradient(src_ptr, rows, cols, dst_ptr);
  else
    ygradient(src_ptr, rows, cols, dst_ptr);
}

} // bpvo
