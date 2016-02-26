#ifndef BPVO_INTERP_UTIL_CV_H
#define BPVO_INTERP_UTIL_CV_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace bpvo {

template <typename T = float>
class BilinearInterpCv
{
 public:
  typedef Eigen::Matrix<T,4,1> Vector4;

 public:
  BilinearInterpCv() {}

  template <class Warp, class PointVector> inline
  void init(const Warp& warp, const PointVector& points, int rows, int cols)
  {
    auto n = points.size();
    _x.resize(n);
    _y.resize(n);
    _valid.resize(n);

    for(size_t i = 0; i < n; ++i)
    {
      auto p = warp(points[i]);
      _x[i] = p.x();
      _y[i] = p.y();

      int xi = (int) _x[i], yi = (int) _y[i];
      _valid[i] = xi>=0 && xi<cols-1 && yi>=0 && yi<rows-1;
    }

    cv::convertMaps(_x, _y, _map1, _map2, CV_16SC2);
  }

  void run(const float* I0_ptr, const cv::Mat_<float>& I1, float* r_ptr) const
  {
    cv::Mat dst;
    cv::remap(I1, dst, _map1, _map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0.0));

    assert( cv::DataType<float>::type == dst.type() && "type mismatch must be float");
    assert( (int) _x.size() == dst.cols && "size mismatch" );

    auto I1_ptr = I1.ptr<float>();
    int n = (int) _x.size();
    for(int i = 0; i < n; ++i) {
      r_ptr[i] = I1_ptr[i] - I0_ptr[i];
    }
  }

  inline const ValidVector& valid() const  { return _valid; }
  inline ValidVector& valid() { return _valid; }

 protected:
  std::vector<float> _x;
  std::vector<float> _y;
  ValidVector _valid;

  cv::Mat _map1, _map2;
}; // BilinearInterpCv

}; // bpvo

#endif // BPVO_INTERP_UTIL_CV_H
