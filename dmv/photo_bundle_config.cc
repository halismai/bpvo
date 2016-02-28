#include <dmv/photo_bundle_config.h>

#include <bpvo/debug.h>
#include <bpvo/config_file.h>
#include <bpvo/utils.h>

#include <iostream>

namespace bpvo {
namespace dmv {

static inline PhotoBundleConfig::SolverType SolverTypeFromString(std::string s)
{
  if(icompare("SparseSchur", s))
    return PhotoBundleConfig::SolverType::SparseSchur;
  else if(icompare("SparseNormalCholesky", s))
    return PhotoBundleConfig::SolverType::SparseNormalCholesky;
  else if(icompare("IterativeSchur", s))
    return PhotoBundleConfig::SolverType::IterativeSchur;
  else {
    Warn("Unknown SolverType '%s'\n", s.c_str());
    return PhotoBundleConfig::SolverType::SparseSchur;
  }
}

static inline PhotoBundleConfig::DescriptorType DescriptorTypeFromString(std::string s)
{
  if(icompare("Patch", s))
    return PhotoBundleConfig::DescriptorType::Patch;
  else if(icompare("BitPlanes", s))
    return PhotoBundleConfig::DescriptorType::BitPlanes;
  else {
    Warn("Unknown DescriptorType '%s'\n", s.c_str());
    return PhotoBundleConfig::DescriptorType::Patch;
  }
}

static inline std::string ToString(PhotoBundleConfig::SolverType s)
{
  switch(s)
  {
    case PhotoBundleConfig::SolverType::SparseNormalCholesky: return "SparseNormalCholesky"; break;
    case PhotoBundleConfig::SolverType::SparseSchur:          return "SparseSchur"; break;
    case PhotoBundleConfig::SolverType::IterativeSchur:       return "IterativeSchur"; break;
  }

  return "Unknown";
}

static inline std::string ToString(PhotoBundleConfig::DescriptorType t)
{
  switch(t)
  {
    case PhotoBundleConfig::DescriptorType::Patch: return "Patch"; break;
    case PhotoBundleConfig::DescriptorType::BitPlanes: return "BitPlanes"; break;
  }

  return "Unknown";
}

PhotoBundleConfig::PhotoBundleConfig(std::string conf_fn)
{
  ConfigFile cf(conf_fn);

  maxIterations = cf.get<int>("maxIterations", 100);
  numThreads = cf.get<int>("numThreads", -1);
  parameterTolerance = cf.get<double>("parameterTolerance", 1e-6);
  functionTolerance = cf.get<double>("functionTolerance", 1e-6);
  gradientTolerance = cf.get<double>("gradientTolerance", 1e-6);
  solverType = SolverTypeFromString(cf.get<std::string>("solverType", "SparseSchur"));
  minZncc = cf.get<double>("minZncc", 0.8);
  descriptorType = DescriptorTypeFromString(cf.get<std::string>("DescriptorType", "Patch"));
  bundleWindowSize = cf.get<int>("bundleWindowSize", 3);
  maxPointsPerImage = cf.get<int>("maxPointsPerImage", 5000);
}

std::ostream& operator<<(std::ostream& os, const PhotoBundleConfig& c)
{
  os << "maxIterations = " << c.maxIterations << "\n";
  os << "numThreads = " << c.numThreads << "\n";
  os << "parameterTolerance = " << c.parameterTolerance << "\n";
  os << "functionTolerance = " << c.functionTolerance << "\n";
  os << "gradientTolerance = " << c.gradientTolerance << "\n";
  os << "SolverType = " << ToString(c.solverType) << "\n";
  os << "DescriptorType = " << ToString(c.descriptorType) << "\n";
  os << "minZncc = " << c.minZncc << "\n";
  os << "bundleWindowSize = " << c.bundleWindowSize << "\n";
  os << "maxPointsPerImage = " << c.maxPointsPerImage;

  return os;
}

} // dmv
} // bpvo

