#ifndef BPVO_UTILS_CV_CEREAL_H
#define BPVO_UTILS_CV_CEREAL_H
#if defined(WITH_CEREAL)
#include <opencv2/core/core.hpp>
// mostly from here http://www.patrikhuber.ch/blog/6-serialising-opencv-matrices-using-boost-and-cereal

namespace cereal {

template <class Archive> inline
void save(Archive& ar, const cv::Mat& m)
{
  int32_t rows = m.rows, cols = m.cols, type = m.type();
  bool continuous = m.isContinuous();

  ar(rows, cols, type, continuous);

  if(continuous) {
    ar(binary_data(m.ptr(), rows*cols*m.elemSize()));
  } else {
    for(int i = 0; i < rows; ++i)
      ar(binary_data(m.ptr(i), cols*m.elemSize()));
  }
}

template <class Archive> inline
void load(Archive& ar, cv::Mat& m)
{
  int32_t rows, cols, type;
  bool continuous;

  ar(rows, cols, type, continuous);
  m.create(rows, cols, type);

  if(continuous) {
    ar(binary_data(m.ptr(), rows*cols*m.elemSize()));
  } else {
    for(int i = 0; i < rows; ++i)
      ar(binary_data(m.ptr(i), cols*m.elemSize()));
  }
}

}; // cereal

#endif // WITH_CEREAL
#endif // BPVO_UTILS_CV_CEREAL_H
