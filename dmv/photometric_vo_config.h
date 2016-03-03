#ifndef BPVO_DMV_PHOTOMETRIC_VO_CONFIG_H
#define BPVO_DMV_PHOTOMETRIC_VO_CONFIG_H

#include <iosfwd>

namespace bpvo {
namespace dmv {

//
//
//
struct PhotometricVoConfig
{
  float minWeight = 0.5;    // minimum pixel weight to use

  //
  // intensity stuff
  //
  double intensityScale = 1.0; //< scale intensity by this factor
  bool withSpatialWeighting = true; //< spatial weithing
  int patchRadius = 1; //< radius of the image patc

  //
  // optimization
  //
  int numPyramidLevels = 4;
  int maxIterations = 50;
  double parameterTolerance = 1e-6;
  double functionTolerance = 1e-6;

  friend std::ostream& operator<<(std::ostream&, const PhotometricVoConfig&);
}; // PhotometricVoConfig


}; // dmv
}; // bpvo

#endif // BPVO_DMV_PHOTOMETRIC_VO_CONFIG_H
