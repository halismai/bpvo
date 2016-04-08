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

#include "bpvo/pose_estimator_params.h"
#include <iostream>

namespace bpvo {

PoseEstimatorParameters::PoseEstimatorParameters(const AlgorithmParameters& p)
    : maxIterations(p.maxIterations)
        , functionTolerance(p.functionTolerance)
        , parameterTolerance(p.parameterTolerance)
        , gradientTolerance(p.gradientTolerance)
        , lossFunction(p.lossFunction)
        , verbosity(p.verbosity) {}


void PoseEstimatorParameters::relaxTolerance(int max_it, float scale_by)
{
  maxIterations = std::min(maxIterations, max_it);
  functionTolerance *= scale_by;
  parameterTolerance *= scale_by;
  gradientTolerance *= scale_by;

  // huber is smoother than tukey, so it takes less iterations
  // but if no weighting was requested, we don't mess with it
  if(lossFunction != LossFunctionType::kL2)
    lossFunction = LossFunctionType::kHuber;
}

std::ostream& operator<<(std::ostream& os, const PoseEstimatorParameters& p)
{
  os << "maxIterations: " << p.maxIterations << "\n";
  os << "functionTolerance: " << p.functionTolerance << "\n";
  os << "parameterTolerance: " << p.parameterTolerance << "\n";
  os << "gradientTolerance: " << p.gradientTolerance << "\n";
  os << "lossFunction: " << ToString(p.lossFunction) << "\n";
  os << "verbosity: " << ToString(p.verbosity);
  return os;
}

}; // bpvo

