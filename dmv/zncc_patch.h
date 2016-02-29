#ifndef DMV_ZNCC_PATCH_H
#define DMV_ZNCC_PATCH_H

#include <bpvo/utils.h>
#include <bpvo/types.h>

#include <dmv/patch_util.h>

#include <Eigen/Core>

#include <iostream>
#include <algorithm>

#include <opencv2/core/core.hpp>

namespace bpvo {
namespace dmv {

template <size_t R, typename T = float>
class ZnccPatch
{
 public:
  typedef T DataType;
  static constexpr int Radius = R;
  static constexpr int Length = GetPatchLength<R>();
  static constexpr int NumBytes = RoundUpTo<Length,16>();

  typedef Eigen::Matrix<T, Length, 1> EigenType;
  typedef Eigen::Map<const EigenType, Eigen::Aligned> EigenMap;

 public:
  inline ZnccPatch() {}

  inline explicit ZnccPatch(const cv::Mat& I, const ImagePoint& p) { set(I, p); }

  inline const ZnccPatch& set(const cv::Mat& I, const ImagePoint& p)
  {
    assert( I.type() == cv::DataType<uint8_t>::type && "image must be uint8_t" );

    extractPatch(I.ptr<uint8_t>(), I.step/I.elemSize1(), I.rows, I.cols,
                 p.y(), p.x(), Radius, _data);

    double m = std::accumulate(_data, _data + Length, 0.0) * (1.0 / Length);
    double dp = 0.0;
    for(int i = 0; i < Length; ++i)
    {
      _data[i] = _data[i] - m;
      dp += (_data[i] * _data[i]);
    }

    _norm = std::sqrt(dp);

    return *this;
  }

  inline double zncc(const ZnccPatch& other) const
  {
    constexpr double eps = 1e-6;
    return _norm > eps && other._norm > eps ?
        (EigenMap(_data).dot(EigenMap(other._data))) / (_norm * other._norm) : -1.0;
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

