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

#include "bpvo/dense_descriptor.h"
#include "bpvo/intensity_descriptor.h"
#include "bpvo/bitplanes_descriptor.h"
#include "bpvo/imgproc.h"
#include "bpvo/utils.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace bpvo {

DenseDescriptor::~DenseDescriptor() {}

DenseDescriptor* DenseDescriptor::Create(const AlgorithmParameters& p, int pyr_level)
{
  switch(p.descriptor)
  {
    case DescriptorType::kIntensity:
      {
        return new IntensityDescriptor();
      } break;

    case DescriptorType::kBitPlanes:
      {
        return new BitPlanesDescriptor(p.sigmaPriorToCensusTransform,
                                       pyr_level >= p.maxTestLevel ? p.sigmaBitPlanes : -1.0f);
      } break;

    default:
      THROW_ERROR("unknown DescriptorType\n");
  }
}

void DenseDescriptor::computeSaliencyMap(cv::Mat& dst) const
{
  dst.create( this->getChannel(0).size(), cv::DataType<float>::type );
  cv::Mat_<float>& d = (cv::Mat_<float>&) dst;

  gradientAbsoluteMagnitude(this->getChannel(0), d);
  for(int i = 1; i < this->numChannels(); ++i)
    gradientAbsoluteMagnitudeAcc(this->getChannel(i), dst.ptr<float>());
}

}; // bpvo

