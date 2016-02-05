#ifndef BPVO_MESTIMATOR_H
#define BPVO_MESTIMATOR_H

#include <bpvo/types.h>
#include <vector>

namespace bpvo {

/**
 */
void computeWeights(LossFunctionType, const std::vector<float>& residuals, float sigma,
                    std::vector<float>& weights);

/**
 * computes the weights for valid points only, invalid points are assigned
 * zero weight
 */
void computeWeights(LossFunctionType, const std::vector<float>& residuals,
                    const std::vector<uint8_t>& valid, float sigma,
                    std::vector<float>& weights);

/**
 * Estimates the scale of the data using robust standard deviation. We also keep
 * the change in the estimated scale across iterations so that time is not
 * wasted if the scale is stable
 */
class AutoScaleEstimator
{
 public:
  AutoScaleEstimator(float tol = 1e-4);

  void reset();
  float getScale() const;

  /**
   */
  float estimateScale(const std::vector<float>& residuals,
                      const std::vector<uint8_t>& valid);

 private:
  float _scale = 1.0, _delta_scale = 1e10, _tol = 1e-4;
  std::vector<float> _buffer;
}; // AutoScaleEstimator

}; // bpvo

#endif // BPVO_MESTIMATOR_H
