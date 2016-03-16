#ifndef BPVO_ROBUST_FIT_H
#define BPVO_ROBUST_FIT_H

#include <bpvo/types.h>

namespace bpvo {

class RobustFit
{
 public:
  /**
   * \param loss_fn the loss function to use
   * \param tune the tuning constant for the loss function. If negative, it will
   *             be set automatically
   */
  RobustFit(LossFunctionType loss_fn, float tune = -1.0);

  /**
   * estimates the scale (the standard deviation) using only valid vectors
   */
  float estimateScale(const ResidualsVector&, const ValidVector&v);

  /**
   * computes the weights. If scale is negative it will be computed using
   * estimateScale
   */
  void computeWeights(const ResidualsVector&, const ValidVector&,
                      WeightsVector&, float scale = -1.0f);

 private:
  LossFunctionType _loss;
  float _tuning_constant;
  ResidualsVector _buffer;

 private:
  float weight(float) const;
}; // RobustFit

}; // bpvo

#endif // BPVO_ROBUST_FIT_H
