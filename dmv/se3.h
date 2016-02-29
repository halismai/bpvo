#ifndef DMV_SE3_H
#define DMV_SE3_H

#if defined(WITH_SOPHUS)
#include <sophus/se3.hpp>

namespace dmv {

template <typename T> using Se3_ = Sophus::SE3Group<T>;
template <typename T> using Mat33_ = Eigen::Matrix<T,3,3>;
template <typename T> using Mat44_ = Eigen::Matrix<T,4,4>;
template <typename T> using Vec4_ = Eigen::Matrix<T,4,1>;
template <typename T> using Vec3_ = Eigen::Matrix<T,3,1>;
template <typename T> using Vec2_ = Eigen::Matrix<T,2,1>;

}; // dmv

#endif

#endif // DMV_SE3_H
