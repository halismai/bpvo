#ifndef BPVO_UTILS_STEREO_CALIBRATION_H
#define BPVO_UTILS_STEREO_CALIBRATION_H

#include <iosfwd>
#include <bpvo/types.h>

namespace bpvo {

struct Calibration
{
  virtual ~Calibration();

  virtual const Matrix33& getIntrinsics() const = 0;
  virtual float getBaseline() const = 0;
}; // Calibration

struct StereoCalibration : public Calibration
{
  inline StereoCalibration(const Matrix33& K_ = Matrix33::Identity(),
                           float baseline_ = 1.0f)
      : K(K_), baseline(baseline_) {}

  inline virtual ~StereoCalibration() {}

  const Matrix33& getIntrinsics() const { return K; }
  float getBaseline() const { return baseline; }

  void scale(double s);

  Matrix33 K;
  float baseline;

  friend std::ostream& operator<<(std::ostream&, const StereoCalibration&);
}; // StereoCalibration


}; // bpvo

#endif // BPVO_UTILS_STEREO_CALIBRATION_H
