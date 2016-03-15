/*
   This file is part of bpvo.

   bpvo is free software: you can redistribute it and/or modify
   it under the terms of the Lesser GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   bpvo is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   Lesser GNU General Public License for more details.

   You should have received a copy of the Lesser GNU General Public License
   along with bpvo.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Contributor: halismai@cs.cmu.edu
 */

#ifndef BPVO_TYPES_H
#define BPVO_TYPES_H

#include <Eigen/Core>

#include <memory>
#include <vector>
#include <iosfwd>
#include <string>

#include <bpvo/aligned_allocator.h>

namespace bpvo {

template <typename _T> using
SharedPointer = std::shared_ptr<_T>;

template <typename _T> using
UniquePointer = std::unique_ptr<_T>;

template <class _T, class ... Args> inline
UniquePointer<_T> make_unique(Args&& ... args) {
#if __cplusplus > 201103L
  return std::make_unique<_T>(std::forward<Args>(args)...);
#else
  return UniquePointer<_T>(new _T(std::forward<Args>(args)...));
#endif
}

template <class _T, class ... Args> inline
SharedPointer<_T> make_shared(Args&& ... args) {
  return std::make_shared<_T>(std::forward<Args>(args)...);
}

#if defined(__AVX__)
static constexpr int DefaultAlignment = 32;
#else
static constexpr int DefaultAlignment = 16;
#endif

template <typename T>
struct AlignedVector
{
  typedef AlignedAllocator<T, DefaultAlignment> allocator_type;
  typedef std::vector<T, allocator_type> type;
}; // AlignedVector

typedef typename AlignedVector<uint8_t>::type ValidVector;
typedef typename AlignedVector<float>::type   ResidualsVector;
typedef ResidualsVector      WeightsVector;

/**
 * 3D points are represented as 4-vectors
 */
typedef Eigen::Matrix<float, 4, 1> Point;

/**
 * Points in the image are 2D
 */
typedef Eigen::Matrix<float, 2, 1> ImagePoint;

/**
 * 3x3 Matrix (e.g. camera intrinsics)
 */
typedef Eigen::Matrix<float, 3, 3> Matrix33;

/**
 * 4x4 matrix (e.g. camera pose)
 */
typedef Eigen::Matrix<float, 4, 4> Matrix44;

/**
 * 3x3 matrix
 */
typedef Eigen::Matrix<float, 3, 4> Matrix34;

/**
 * The pose is also a 4x4 matrix
 */
typedef Matrix44 Pose;

/**
 */
typedef Eigen::Matrix<float,6,6> PoseCovariance;


/**
 * defines an STL container type suitable for Eigen. This is unnecessary with
 * c++11
 */
template <class MatrixType, template<class, class> class Container = std::vector>
struct EigenAlignedContainer
{
  typedef Eigen::aligned_allocator<MatrixType>  allocator_type;
  typedef Container<MatrixType, allocator_type> type;
}; // EigenAlignedContainer

/**
 * Supported types of the robust function to use during IRLS
 */
enum LossFunctionType
{
  kHuber = 0x10,
  kTukey,
  kL2 // not robust (ordinary least squares)
}; // LossFunctionType

enum VerbosityType
{
  kIteration = 0x20, //< print details of every iteration
  kFinal,     //< show a brief summary at the end
  kSilent,    //< say nothing
  kDebug      //< print debug info
}; // VerbosityType

enum DescriptorType
{
  kIntensity = 0x30, //< raw intensity (fast)
  kBitPlanes         //< bit-planes (robust)
}; // DescriptorType

struct AlgorithmParameters
{
  //
  // general algorithm configurations
  //

  /**
   * number of pyramid levels (negative means auto)
   * */
  int numPyramidLevels;

  /**
   * std. deviation of Gaussian to apply to the image before computing the
   * census transform
   *
   * A negative (or zero) value means none
   */
  float sigmaPriorToCensusTransform;

  /**
   * std. deviation of a Gaussian to smooth the Bit-Planes
   */
  float sigmaBitPlanes;


  //
  // optimization
  //

  /**
   * maximum number of iterations
   */
  int maxIterations;

  /**
   * tolerance on the norm of the parameter vector
   */
  float parameterTolerance;

  /**
   * tolerance on the function value
   */
  float functionTolerance;

  /**
   * tolerance on the gradient of the objective J'*F
   */
  float gradientTolerance;

  /**
   * If true, converge thresholds will be reduced for coarse pyramid levels
   */
  bool relaxTolerancesForCoarseLevels;

  /**
   * The loss function to use
   */
  LossFunctionType lossFunction;

  /**
   * Descriptor type
   */
  DescriptorType descriptor;

  /**
   * how much verbosity you want!
   */
  VerbosityType verbosity;

  //
  // KeyFraming
  //

  /**
   * We set a new keyframe if the estimated translation is larger than this
   * amount
   *
   * Unit: meters
   */
  float minTranslationMagToKeyFrame;

  /**
   * We set a new keyframe if rotation exceeds this amount
   *
   * Units: degrees
   *
   * This test is done by converting the rotation matrix to Euler angles and
   * using the Inf norm
   */
  float minRotationMagToKeyFrame;


  /**
   * If the fraction of good points drops below this threshold we keyframe
   *
   * Good points are the ones that project onto the image and their associated
   * weight from IRLS is greater than a threhsold
   */
  float maxFractionOfGoodPointsToKeyFrame;

  /**
   * Threhsold on the IRLS weights to consider a point `good'
   *
   * IRLS weights are normalized to [0,1] with 0 being bad
   */
  float goodPointThreshold;


  //
  // pixel selection
  //

  /**
   * Minimum number of pixels to do non-maxima suppression
   */
  int minNumPixelsForNonMaximaSuppression;

  /**
   */
  int nonMaxSuppRadius;

  /**
   * Minimum number of pixels to estimate pose
   */
  int minNumPixelsToWork;

  /**
   * Minimum saliency value for a pixel to be used in the optimization
   */
  float minSaliency;

  /**
   * minimum valid disparity to use
   */
  float minDisparity;

  /**
   * maximum disparity to use
   */
  float maxDisparity;

  //
  // other
  //

  /**
   * TODO
   */
  int maxTestLevel;


  /**
   * normalize the values going into the linear system. Produces a solution
   * faster
   */
  bool withNormalization;

  /**
   * Sets default parameters
   */
  AlgorithmParameters();

  /**
   * Loads the parameters from file
   */
  explicit AlgorithmParameters(std::string filename);

  /**
   * stream insertion
   */
  friend std::ostream& operator<<(std::ostream&, const AlgorithmParameters&);
}; // AlgorithmParameters


/**
 * Tells you why the code returned a certain result
 */
enum PoseEstimationStatus
{
  kParameterTolReached = 0x30, //< optimizer reached the desired parameter tolerance
  kFunctionTolReached,  //< ditto function value
  kGradientTolReached,  //< ditto gradient value (J'*F)
  kMaxIterations,       //< Maximum number of iteration
  kSolverError          //< !
}; // PoseEstimationStatus


/**
 * tells you why the code decided to keyframe
 */
enum KeyFramingReason
{
  kLargeTranslation = 0x40,      //< translation was large
  kLargeRotation,         //< rotation was large
  kSmallFracOfGoodPoints, //< fraction of good points is low
  kNoKeyFraming,          //< there was no keyframe
  kFirstFrame
}; // KeyFramingReason


/**
 * Statistics from the optimizer
 */
struct OptimizerStatistics
{
  /**
   */
  int numIterations;

  /**
   * error at the last iteration of the optimization.
   * This is the squared norm of the weighted residuals vector
   */
  float finalError;

  /**
   * first order optimiality at the end of the optimization
   */
  float firstOrderOptimality;

  /**
   *
   */
  PoseEstimationStatus status;

  friend std::ostream& operator<<(std::ostream& os, const OptimizerStatistics&);
}; // OptimizerStats

class PointCloud;

/**
 * Output from VO include the pose and other useful statistics
 */
struct Result
{
  /**
   * The estimated pose.
   *
   * This is relative motion of the most recent frame
   */
  Pose pose;

  /**
   * Covariance of the estimated parameters
   */
  PoseCovariance covariance;

  /**
   * opeimizer statistics from every iteration. The first element corresponds to
   * the `finest' pyramid level (highest resolution)
   */
  std::vector<OptimizerStatistics> optimizerStatistics;

  /**
   * If this is set to 'true' then we set a keyframe (usually from the previous
   * image)
   */
  bool isKeyFrame;

  /**
   * reason for keyframeing
   */
  KeyFramingReason keyFramingReason;

  /**
   * Point cloud and its own.
   * check the pointer before using it, it is not null iff we have points (based
   * on keyframing)
   *
   * Before using the point cloud, it must also be transformed with the
   * associated pose
   */
  UniquePointer<PointCloud> pointCloud;

  /**
   * stream insertion
   */
  friend std::ostream& operator<<(std::ostream&, const Result&);

  Result();
  Result(Result&&);
  Result(const Result&) = delete;

  Result& operator=(Result&&);

  ~Result();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // Result


/**
 * The image size.
 *
 * All images are rows x cols
 * The stride (number of elements per columns) is the same as the 'cols'
 */
struct ImageSize
{
  int rows = 0;
  int cols = 0;

  inline ImageSize(int r = 0, int c = 0) : rows(r), cols(c) {}

  inline int numel() const { return rows*cols; }

  friend std::ostream& operator<<(std::ostream&, const ImageSize&);
}; // ImageSize


std::string ToString(LossFunctionType);
std::string ToString(VerbosityType);
std::string ToString(PoseEstimationStatus);
std::string ToString(KeyFramingReason);

LossFunctionType LossFunctionTypeFromString(std::string);
DescriptorType DescriptorTypeFromString(std::string);
VerbosityType VerbosityTypeFromString(std::string);

}; // bpvo

#endif // BPVO_TYPES_H

