#include "utils/stereo_calibration.h"
#include <iostream>

namespace bpvo {

Calibration::~Calibration() {}

void StereoCalibration::scale(double s)
{
  if(s > 1.0) {
    this->K *= (1.0 / s); this->K(2,2) = 1.0;
    this->baseline *= s;
  }
}

std::ostream& operator<<(std::ostream& os, const StereoCalibration& c)
{
  os << c.K << "\n";
  os << "baseline: " << c.baseline;

  return os;
}

} // bpvo
