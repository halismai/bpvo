#include <bpvo/debug.h>

#if defined(WITH_CEREAL)

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>

#include <bpvo/types.h>
#include <bpvo/utils.h>

#include <utils/eigen_cereal.h>
#include <utils/cv_cereal.h>

#include <fstream>

#include <string.h>

template <class T> inline
void save(std::string filename, const T& M)
{
  std::ofstream ofs(filename, std::ios::binary);
  cereal::BinaryOutputArchive ar(ofs);
  ar(M);
}

template <class T> inline
void load(std::string filename, T& M)
{
  std::ifstream ifs(filename, std::ios::binary);
  cereal::BinaryInputArchive ar(ifs);
  ar(M);
}

int main()
{
  {
    Eigen::MatrixXf M = Eigen::MatrixXf::Random(3,4);

    save("/tmp/matrix", M);
    std::cout << "Archived:\n" << M << "\n\n";

    Eigen::MatrixXf M2;
    load("/tmp/matrix", M2);
    std::cout << "Loaded:\n" << M2 << "\n\n";

    THROW_ERROR_IF(memcmp(M.data(), M2.data(), M.rows()*M.cols()*sizeof(*M.data())), "Cereal failed");
  }

  {
    Eigen::Matrix<int,3,4> M = Eigen::Matrix<int,3,4>::Identity();
    save("/tmp/matrix", M);
    std::cout << "Archived:\n" << M << "\n\n";

    decltype(M) M2;
    load("/tmp/matrix", M2);
    std::cout << "Loaded:\n" << M2 << "\n\n";

    THROW_ERROR_IF(memcmp(M.data(), M2.data(), M.rows()*M.cols()*sizeof(*M.data())), "Cereal failed");
  }

  {
    cv::Mat M(4, 5, CV_8U);
    cv::randu(M, cv::Scalar(0), cv::Scalar(255));
    save("/tmp/image", M);

    cv::Mat M2;
    load("/tmp/image", M2);
    THROW_ERROR_IF(memcmp(M.ptr(), M2.ptr(), M.rows*M.cols*M.elemSize()), "Cereal failed");

    M.create(8, 10, CV_MAKETYPE(cv::DataType<float>::type, 4));
    cv::randu(M, 0, 255);
    save("/tmp/image", M); load("/tmp/image", M2);
    THROW_ERROR_IF(memcmp(M.ptr(), M2.ptr(), M.rows*M.cols*M.elemSize()), "Cereal failed");

    printf("size %zu\n", M.rows*M.cols*M.elemSize());
  }

  printf("all good\n");

  return 0;
}

#else
int main() { Fatal("compile WITH_CEREAL"); }
#endif
