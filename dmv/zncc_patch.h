#ifndef DMV_ZNCC_PATCH_H
#define DMV_ZNCC_PATCH_H

#include <dmv/patch.h>
#include <Eigen/Core>

#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

namespace bpvo {
namespace dmv {

template <size_t R, typename T = float>
class ZnccPatch
{

  typedef Eigen::Matrix<T, GetPatchLength<R>(), 1> EigenType;
  typedef Eigen::Map<const EigenType, Eigen::Aligned> EigenMap;

 public:
  typedef T DataType;
  static constexpr int Radius = R;

 public:
  ZnccPatch() {}

  inline const ZnccPatch& set(const cv::Mat& I, const ImagePoint& p)
  {
    auto N = GetPatchLength<Radius>();
    cv::Size siz(N, N);

    cv::Mat _buffer;
    cv::getRectSubPix(I, siz, cv::Point2f(p.x(), p.y()), _buffer, cv::DataType<DataType>::type);

    auto m = cv::sum(_buffer)[0] / (float) (_buffer.rows * _buffer.cols);

    const auto ptr = _buffer.template ptr<DataType>();
    for(size_t i = 0; i < N; ++i)
      _data[i] = cv::saturate_cast<DataType>(ptr[i] - m);

    _norm = std::sqrt(static_cast<double>(EigenMap(_data).dot(EigenMap(_data))));

    for(size_t i = N; i < sizeof(_data) / sizeof(DataType); ++i)
      _data[i] = DataType(0);

    return *this;
  }

  inline double zncc(const ZnccPatch& other) const
  {
    constexpr double eps = 1e-3;
    return _norm > eps && other._norm > eps ?
        (EigenMap(_data).dot(EigenMap(other._data))) /
        static_cast<double>(_norm * other._norm) : -1.0;
  }

  inline const DataType* data() const { return _data; }

  inline double norm() const { return _norm; }

 protected:
  alignas(16) DataType _data[ RoundUpTo<GetPatchLength<Radius>(),16>() ];
  double _norm = 0.0;
}; // ZnccPatch

}; // dmv
}; // bpvo

#endif // DMV_ZNCC_PATCH_H

