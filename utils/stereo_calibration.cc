#include "utils/stereo_calibration.h"
#include <iostream>

namespace bpvo {

Calibration::~Calibration() {}

std::ostream& operator<<(std::ostream& os, const StereoCalibration& c)
{
  os << c.K << "\n";
  os << "baseline: " << c.baseline;

  return os;
}

} // bpvo
