#ifndef BPVO_TYPES_H
#define BPVO_TYPES_H

#include <Eigen/Core>

#include <memory>
#include <vector>
#include <iosfwd>
#include <string>

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
  kHuber,
  kTukey,
  kL2 // not robust (ordinary least squares)
}; // LossFunctionType

enum VerbosityType
{
  kIteration, //< print details of every iteration
  kFinal,     //< show a brief summary at the end
  kSilent,    //< say nothing
  kDebug      //< print debug info
}; // VerbosityType

struct AlgorithmParameters
{
  /**
   * minimum number of pixels to select (instead of dense)
   */
  static const int MIN_NUM_FOR_PIXEL_PSELECTION = 320*240;

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
  kParameterTolReached, //< optimizer reached the desired parameter tolerance
  kFunctionTolReached,  //< ditto function value
  kGradientTolReached,  //< ditto gradient value (J'*F)
  kMaxIterations,       //< Maximum number of iteration
  kSolverError         //< !
}; // PoseEstimationStatus


/**
 * tells you why the code decided to keyframe
 */
enum KeyFramingReason
{
  kLargeTranslation,      //< translation was large
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
   * optimizer statistics from every iteration. The first element corresponds to
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
   * stream insertion
   */
  friend std::ostream& operator<<(std::ostream&, const Result&);

  Result();

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
}; // ImageSize


std::string ToString(LossFunctionType);
std::string ToString(VerbosityType);
std::string ToString(PoseEstimationStatus);
std::string ToString(KeyFramingReason);

LossFunctionType LossFunctionTypeFromString(std::string);
VerbosityType VerbosityTypeFromString(std::string);

}; // bpvo

#endif // BPVO_TYPES_H

