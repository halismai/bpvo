#ifndef BPVO_OPENCV_H
#define BPVO_OPENCV_H

#include <bpvo/types.h>
#include <opencv2/core/core.hpp>

namespace bpvo {

template <typename T> inline
cv::Mat ToOpenCV(const T* p, int rows, int cols)
{
  return cv::Mat(rows,cols, cv::DataType<T>::type, (void*) p);
}

template <typename T> inline
cv::Mat ToOpenCV(const T* p, const ImageSize& siz)
{
  return ToOpenCV(p, siz.rows, siz.cols);
}

}; // bpvo

#endif

