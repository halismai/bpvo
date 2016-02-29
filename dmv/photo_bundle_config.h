#ifndef BPVO_DMV_BUNDLE_DATA_CONFIG_H
#define BPVO_DMV_BUNDLE_DATA_CONFIG_H

#include <iosfwd>
#include <string>

namespace bpvo {
namespace dmv {

struct PhotoBundleConfig
{
  inline PhotoBundleConfig() {}
  explicit PhotoBundleConfig(std::string);

  enum class SolverType
  {
    SparseNormalCholesky,
    SparseSchur,
    IterativeSchur
  }; // SolverType

  enum class DescriptorType
  {
    Patch,
    BitPlanes
  }; // DescriptorType

  //
  // optimization options
  //
  int maxIterations = 100;
  int numThreads = -1; // negative is AUTO
  double parameterTolerance = 1e-6;
  double functionTolerance = 1e-6;
  double gradientTolerance = 1e-6;
  SolverType solverType = SolverType::SparseSchur;
  DescriptorType descriptorType = DescriptorType::Patch;

  double minZncc = 0.8;

  int bundleWindowSize = 3;
  int maxFrameDistance = 3;
  int maxPointsPerImage = 5000;

  friend std::ostream& operator<<(std::ostream, const PhotoBundleConfig&);
}; // PhotoBundleConfig

}; // dmv
}; // bpvo

#endif // BPVO_DMV_BUNDLE_DATA_CONFIG_H
