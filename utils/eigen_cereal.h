#ifndef BPVO_UTILS_EIGEN_CEREAL_H
#define BPVO_UTILS_EIGEN_CEREAL_H
#if defined(WITH_CEREAL)

// mostly from here http://stackoverflow.com/questions/22884216/serializing-eigenmatrix-using-cereal-library

#include <Eigen/Dense>
#include <cereal/cereal.hpp>

namespace cereal {

template <class Archive, class Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols> inline
typename std::enable_if<traits::is_output_serializable<BinaryData<Scalar>,Archive>::value, void>::type
save(Archive& ar, const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& m)
{
  int32_t rows = m.rows();
  int32_t cols = m.cols();
  ar(rows);
  ar(cols);
  ar(binary_data(m.data(), rows*cols*sizeof(Scalar)));
}

template <class Archive, class Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols> inline
typename std::enable_if<traits::is_input_serializable<BinaryData<Scalar>,Archive>::value, void>::type
load(Archive& ar, Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& m)
{
  int32_t rows;
  int32_t cols;

  ar(rows);
  ar(cols);
  m.resize(rows, cols);
  ar(binary_data(m.data(), static_cast<std::size_t>(rows*cols*sizeof(Scalar))));
}


}; // cereal

#endif // WITH_CEREAL
#endif // BPVO_UTILS_EIGEN_CEREAL_H
