#ifndef BPVO_IMGPROC_H
#define BPVO_IMGPROC_H

#include <opencv2/core/core.hpp>
#include <bpvo/debug.h>

namespace bpvo {

/**
 * allows to subsample the disparities using a pyramid level
 */
struct DisparityPyramidLevel
{
  DisparityPyramidLevel(const cv::Mat& D, int pyr_level)
      : _D_ptr(D.ptr<float>()), _stride(D.cols), _scale(1 << pyr_level) {}

  FORCE_INLINE float operator()(int r, int c) const
  {
    return *(_D_ptr + index(r,c));
  }

  FORCE_INLINE int index(int r, int c) const {
    return (r*_scale) * _stride + c*_scale;
  }

  const float* _D_ptr;
  int _stride;
  int _scale;
}; // DisparityPyramidLevel

/**
 */
template <class T>
struct IsLocalMax
{
  inline IsLocalMax() {}

  inline IsLocalMax(const T* ptr, int stride, int radius)
      : _ptr(ptr), _stride(stride), _radius(radius)
  {
    assert( _stride > 0);
  }

  inline void setStride(int s) { _stride = s; }
  inline void setRadius(int r) { _radius = r; }
  inline void setPointer(const T* p) { _ptr = p; }

  /**
   * \return true if element at location (row,col) is a strict local maxima
   * within the specified radius
   */
  FORCE_INLINE bool operator()(int row, int col) const
  {
    if(_radius > 0) {
      auto v = _ptr[row*_stride + col];
      for(int r = -_radius; r <= _radius; ++r)
        for(int c = -_radius; c <= _radius; ++c)
          if(!(!r && !c) && _ptr[(row+r)*_stride + col + c] >= v)
            return false;
    }
    return true;
  }

 private:
  const T* _ptr;
  int _stride, _radius;
}; // IsLocalMax

template <class SaliencyMapT>
class ValidPixelPredicate
{
  typedef typename SaliencyMapT::value_type DataType;
  static constexpr DataType MinSaliency = 0.01;
  static constexpr float MinDisparity = 0.30f;

 public:
  inline ValidPixelPredicate(const DisparityPyramidLevel& dmap,
                             const SaliencyMapT& smap, int nms_radius)
      : _dmap(dmap), _smap(smap),
      _is_local_max(smap.template ptr<DataType>(), smap.cols, nms_radius) {}

  FORCE_INLINE bool operator()(int r, int c) const
  {
    return _dmap(r,c) > MinDisparity && _smap(r,c) > MinSaliency && _is_local_max(r,c);
  }

 protected:
  const DisparityPyramidLevel& _dmap;
  const SaliencyMapT& _smap;
  IsLocalMax<DataType> _is_local_max;
}; // ValidPixelPredicate

}; // bpvo

#endif // BPVO_IMGPROC_H
