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

#include <opencv2/core/core.hpp>

#include <bpvo/vo_pose.h>
#include <bpvo/utils.h>
#include <bpvo/intensity_descriptor.h>

#include <cmath>

namespace bpvo {

static int getNumberOfPyramidLevels(int n_levels, int min_image_dim)
{
  return n_levels < 0 ?
      1 + std::round(std::log2(min_image_dim / (double) n_levels)) :
      n_levels;
}

template <typename T> static inline
cv::Mat ToOpenCV(const T* p, const ImageSize& siz)
{
  return cv::Mat(siz.rows, siz.cols, cv::DataType<T>::type, (void*) p);
}

VisualOdometryPose::VisualOdometryPose(const Matrix33& K, const float b, ImageSize image_size,
                                       AlgorithmParameters p)
  : _image_size(image_size)
  , _params(p)
  , _pose_est_params(p)
  , _pose_est_params_low_res(p)
  , _image_pyramid( getNumberOfPyramidLevels(p.numPyramidLevels, 40) )
  , _tdata_pyr( _image_pyramid.size() )
{
  _params.numPyramidLevels = _image_pyramid.size();

  if(_params.relaxTolerancesForCoarseLevels)
    _pose_est_params_low_res.relaxTolerance();

  Matrix33 K_pyr(K);
  float b_pyr = b;
  _tdata_pyr[0] = make_unique<TemplateData>(0, K_pyr, b_pyr, _params);
  for(size_t i = 1; i < _tdata_pyr.size(); ++i) {
    K_pyr *= 0.5; K_pyr(2,2) = 1.0;
    b_pyr *= 2.0;
    _tdata_pyr[i] = make_unique<TemplateData>(i, K_pyr, b_pyr, _params);
  }
}

static inline UniquePointer<DenseDescriptor> makeDescriptor(DescriptorType t)
{
  if(DescriptorType::kIntensity == t)
    return UniquePointer<DenseDescriptor>(new IntensityDescriptor);
  else
    THROW_ERROR("not implemented");
}

void VisualOdometryPose::setTemplate(const uint8_t* image_ptr, const float* dmap_ptr)
{
  const auto I = ToOpenCV(image_ptr, _image_size);
  const auto D = ToOpenCV(dmap_ptr, _image_size);

  _image_pyramid.compute(I);

  for(size_t i = 0; i < _tdata_pyr.size(); ++i) {
    auto desc = makeDescriptor(_params.descriptor);
    desc->compute(_image_pyramid[i]);
    _tdata_pyr[i]->setData(desc.get(), D);
  }
}


Result VisualOdometryPose::estimatePose(const uint8_t* image_ptr,
                                        const Matrix44& T_init)
{
  Result ret;
  ret.pose = T_init;
  auto& stats = ret.optimizerStatistics;
  stats.resize(_tdata_pyr.size());

  auto desc = makeDescriptor(_params.descriptor);
  _image_pyramid.compute(ToOpenCV(image_ptr, _image_size));

  _pose_estimator.setParameters(_pose_est_params_low_res);
  for(int i = _image_pyramid.size()-1; i >= _params.maxTestLevel; --i) {
    if(i >= _params.maxTestLevel)
    _pose_estimator.setParameters(_pose_est_params); // restore high res params

    if(_tdata_pyr[i]->numPixels() > _params.minNumPixelsToWork) {
      desc->compute(_image_pyramid[i]);
      stats[i] = _pose_estimator.run(_tdata_pyr[i].get(), desc.get(), ret.pose);
    } else {
      Warn("Not enough points at octave %d [needs %d]\n", i, _params.minNumPixelsToWork);
    }
  }

  dprintf("estimatePose done %zu\n", stats.size());
  std::cout << ret << std::endl;
  return ret;
}

bool VisualOdometryPose::hasData() const
{
  return _tdata_pyr[_params.maxTestLevel]->numPoints() > 0;
}

} // bpvo

