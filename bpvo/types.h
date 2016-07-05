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

#if defined(IS_64BIT)
typedef typename AlignedVector<uint16_t>::type ValidVector;
#else
typedef typename AlignedVector<uint8_t>::type ValidVector;
#endif

typedef typename AlignedVector<float>::type    ResidualsVector;
typedef ResidualsVector                        WeightsVector;

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
  kIntensity = 0x30,           //< raw intensity (fast)
  kIntensityAndGradient,       //< intensity + gradient constraint
  kDescriptorFieldsFirstOrder, //< 1-st order descriptor fields
  kDescriptorFieldsSecondOrder,//< 1-nd order df
  kLatch,                      //< The latch descriptor
  kCentralDifference,          //< central diff (see paper)
  kLaplacian,                  //< Laplacian
  kBitPlanes                   //< bit-planes (robust)
}; // DescriptorType

/**
 * Algorithm to use to estimate the gradient of the image
 */
enum GradientEstimationType
{
  kCentralDifference_3, // 3-tap central difference [1 0 -1] / 2
  kCentralDifference_5, // 5-tap central difference [-1 8 0 -8 1] / 18
}; // GradientEstimationType

enum InterpolationType
{
  kLinear,  // linear interpolation
  kCosine,  // Cosine interpolation
  kCubic,   // cubic
  kCubicHermite, // cubic hermite (pchip)
}; // InterpolationType

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
   * minium dimension of the image at the coarest pyramid level. This will be
   * use dif numPyramidLevels <=0
   */
  int minImageDimensionForPyramid;

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

  /**
   * std. deviation of Gaussian to apply before computing the gradient for DF
   */
  float dfSigma1;

  /**
   * std. deviation of a Gaussian to apply ont he DF
   */
  float dfSigma2;

  /**
   * Number of bytes for the latch descriptor
   */
  int latchNumBytes;

  /**
   * rotation invariance for the latch descriptor
   */
  bool latchRotationInvariance;

  /**
   * Half ssd size for the latch descriptor
   */
  int latchHalfSsdSize;

  /**
   * Patch radius to use when computing the central difference descriptor
   */
  int centralDifferenceRadius;

  /**
   * sigma before computing the descriptor
   */
  float centralDifferenceSigmaBefore;

  /**
   * sigma after
   * TODO (we only need a single variable, maybe call it sigmaChannel)
   */
  float centralDifferenceSigmaAfter;

  /**
   * Kernel size for the Laplacian
   */
  int laplacianKernelSize;

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
   * Gradient estimation
   */
  GradientEstimationType gradientEstimation;

  /**
   * Interpolation type for image warping
   */
  InterpolationType interp;

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
   * Radius of non maxima suppression when performing pixel selection over the
   * saliency map of the densee descriptor
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
  float minValidDisparity;

  /**
   * maximum disparity to use
   */
  float maxValidDisparity;


  //
  // other
  //

  /**
   * Maximum level to use for pose estimation in the pyramid. By default this is
   * set to 0, meaning process up to level 0.  Level zero is the finest pyramid
   * level.
   *
   * If you want to run up to half the original resolution, set this value to 1,
   * which corresponds to the second level in the pyramid, etc.
   *
   * numPyramidLevels must enough to accomodate maxTestLevel
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
 * Tells you why the code decided to keyframe.
 *
 * Thresholds for the parameters are decided in AlgorithmParameters
 */
enum KeyFramingReason
{
  kLargeTranslation = 0x40,  //< translation was large
  kLargeRotation,            //< rotation was large
  kSmallFracOfGoodPoints,    //< fraction of good points is low
  kNoKeyFraming,             //< there was no keyframe
  kFirstFrame
}; // KeyFramingReason

/**
 * Statistics from the optimizer
 */
struct OptimizerStatistics
{
  OptimizerStatistics();

  /**
   * Number of iterations it took to converge. If this is unsually high, it may
   * indicate a problem. For example, motion was too large, or the optimizer got
   * stuck in a local minima
   */
  int numIterations;

  /**
   * Error at the last iteration of the optimization.
   * This is the squared norm of the weighted residuals vector. Monitering this
   * across iterations is useful, to access the quality of the solution and/or
   * detect issues with the data.
   */
  float finalError;

  /**
   * First order optimiality at the end of the optimization, aka the Inf norm of
   * the gradient vector at the solution.
   *
   * Sometimes this is called the 1-st order necessary condition for convergece.
   * In plain language, it is gradient of the function.
   *
   * If this is zero, then the 1-st order necessary condition has been
   * satified. However, it is usually possible (especially with the IC)
   * formulation. A large value does not mean a bad solution
   */
  float firstOrderOptimality;

  /**
   *  Status from pose estimation module
   */
  PoseEstimationStatus status;

  friend std::ostream& operator<<(std::ostream& os, const OptimizerStatistics&);
}; // OptimizerStats

class PointCloud;

/**
 * Output from VO including the pose and other useful statistics
 */
struct Result
{
  /**
   * The estimated pose.
   *
   * This is the relative motion of the most recently added frame wrt the
   * current reference/template frame
   */
  Pose pose;

  /**
   * Covariance of the estimated parameters. The inverse of the weighed hessian
   * at the solution. It is a 6x6 matrix with rotations present first.
   */
  PoseCovariance covariance;

  /**
   * Optimizer statistics at every iteration. The first element corresponds to
   * the `finest' pyramid level (highest resolution).
   *
   * You should use optimizerStatistics at the index corresponding to
   * maxTestLevel, i.e.
   *
   * auto stats = result.optimizerStatistics[ params.maxTestLevel ];
   *
   */
  std::vector<OptimizerStatistics> optimizerStatistics;

  /**
   * If this is set to 'true' then we set a keyframe (usually from the previous
   * image)
   *
   * If isKeyFrame := true, then a point cloud may be obtained from VO
   */
  bool isKeyFrame;

  /**
   * Reason for keyframeing
   */
  KeyFramingReason keyFramingReason;

  /**
   * Point cloud from the most recenet keyframe.
   *
   * Check the pointer before using it, as it is not null iff we have points (based
   * on keyframing)
   *
   * Before using the point cloud, it must also be transformed with the
   * associated pose. We return the pointCloud in the local coordinates of the
   * image in case the user wants to do other color extraction/processing.
   *
   * The point cloud is extracted from integer pixel locations, to get the
   * correspoinding pixel in the image you can do:
   *
   *  auto uv = normHomog( K.inverse() * X_i);
   *  int col = std::round(uv[0]);
   *  int row = std::round(uv[1]);
   *
   * You need the round() for floating point errors
   */
  UniquePointer<PointCloud> pointCloud;

  /**
   * stream insertion
   */
  friend std::ostream& operator<<(std::ostream&, const Result&);

  Result();
  Result(Result&&) noexcept;
  Result& operator=(Result&&) noexcept;

  //
  // copy and assigment are deleted. The class must be moved
  //
  Result(const Result&) = delete;
  Result& operator=(const Result&) = delete;

  ~Result();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
}; // Result


/**
 * The image size.
 *
 * All images are rows x cols
 * The stride (number of elements per columns), which is usually the same as the
 * 'cols' for single channel images
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
std::string ToString(DescriptorType);
std::string ToString(GradientEstimationType);
std::string ToString(InterpolationType);

LossFunctionType LossFunctionTypeFromString(std::string);
DescriptorType DescriptorTypeFromString(std::string);
VerbosityType VerbosityTypeFromString(std::string);
GradientEstimationType GradientEstimationTypeFromString(std::string);
InterpolationType InterpolationTypeFromString(std::string);

}; // bpvo

#endif // BPVO_TYPES_H

