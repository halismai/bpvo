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

#ifndef BPVO_POSE_ESTIMATOR_PARAMS_H
#define BPVO_POSE_ESTIMATOR_PARAMS_H

#include <bpvo/types.h>
#include <iosfwd>

namespace bpvo {

struct PoseEstimatorParameters
{
  int maxIterations = 50;
  int maxFuncEvals = 6*200;
  float functionTolerance   = 1e-6;
  float parameterTolerance  = 1e-6;
  float gradientTolerance   = 1e-6;
  LossFunctionType lossFunction = LossFunctionType::kHuber;

  VerbosityType verbosity = VerbosityType::kSilent;

  /**
   */
  inline PoseEstimatorParameters() {}

  /**
   */
  explicit PoseEstimatorParameters(const AlgorithmParameters& p);

  /**
   */
  void relaxTolerance(int max_it = 20, float scale_by = 10.0f);

  /**
   */
  friend std::ostream& operator<<(std::ostream&, const PoseEstimatorParameters&);
}; // PoseEstimatorParameters


}; // bpvo

#endif // BPVO_POSE_ESTIMATOR_PARAMS_H
