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
#include <bpvo/parallel.h>
#include <bpvo/utils.h>
#include <bpvo/intensity_descriptor.h>
#include <cmath>

#define WITH_THREAD_POOL 1
#define WITH_TBB_TASK_GROUP 1

#if !defined(WITH_TBB)
#undef WITH_TBB_TASK_GROUP
#define WITH_TBB_TASK_GROUP 0
#endif

#if WITH_TBB_TASK_GROUP
#include <tbb/task_group.h>
#endif

#if WITH_THREAD_POOL
#include <externals/ThreadPool.h>
#endif

namespace bpvo {

static int getNumberOfPyramidLevels(int min_image_dim, int min_allowed_res = 40)
{
  return 1 + std::round(std::log2(min_image_dim / (double) min_allowed_res));
}

static int getNumberOfPyramidLevels(ImageSize s, int min_allowed_res)
{
  return getNumberOfPyramidLevels(std::min(s.rows, s.cols), min_allowed_res);
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
  , _image_pyramid( _params.numPyramidLevels < 0 ?
                    getNumberOfPyramidLevels(image_size, 40) :
                    _params.numPyramidLevels )
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

static inline UniquePointer<DenseDescriptor> MakeDescriptor(DescriptorType t)
{
  if(DescriptorType::kIntensity == t)
    return UniquePointer<DenseDescriptor>(new IntensityDescriptor);
  else
    THROW_ERROR("not implemented");
}

struct VoPoseSetTemplateBody : public ParallelForBody
{
  VoPoseSetTemplateBody(DescriptorType desc_type, const ImagePyramid& image_pyramid,
                        const cv::Mat& D, std::vector<UniquePointer<TemplateData>>& tdata_pyr)
      : _desc_type(desc_type), _image_pyramid(image_pyramid), _D(D), _tdata_pyr(tdata_pyr)
    {
      assert( _image_pyramid.size() == (int) _tdata_pyr.size() );
    }

  inline void operator()(const Range& range) const
  {
    for(int i = range.begin(); i != range.end(); ++i)
    {
      auto desc = MakeDescriptor(_desc_type);
      desc->compute(_image_pyramid[i]);
      _tdata_pyr[i]->setData( desc.get(), _D);
    }
  }

 protected:
  DescriptorType _desc_type;
  const ImagePyramid& _image_pyramid;
  const cv::Mat& _D;
  std::vector<UniquePointer<TemplateData>>& _tdata_pyr;
}; // VoPoseSetTemplateBody

void VisualOdometryPose::setTemplate(const uint8_t* image_ptr, const float* dmap_ptr)
{
  const auto I = ToOpenCV(image_ptr, _image_size);
  _image_pyramid.compute(I);

  setTemplate(_image_pyramid, dmap_ptr);
}

void VisualOdometryPose::setTemplate(const ImagePyramid& image_pyramid,
                                     const float* dmap_ptr)
{
  const auto D = ToOpenCV(dmap_ptr, _image_size);

#if WITH_TBB_TASK_GROUP
  tbb::task_group tasks;
  for(int i = 0; i < image_pyramid.size(); ++i)
    tasks.run([=]()
              {
              auto desc = MakeDescriptor(_params.descriptor);
              desc->compute(_image_pyramid[i]);
              _tdata_pyr[i]->setData(desc.get(), D);
              });
  tasks.wait();
#elif WITH_THREAD_POOL
  ThreadPool pool(4);
  std::vector<std::future<void>> results;
  for(int i = 0; i < image_pyramid.size(); ++i)
    results.emplace_back(
        pool.enqueue(
            [=]()
            {
              auto desc = MakeDescriptor(_params.descriptor);
              desc->compute(image_pyramid[i]);
              _tdata_pyr[i]->setData(desc.get(), D);
            }
            )
        );
  for(auto&& r : results)
    r.get();
#else
  VoPoseSetTemplateBody func(_params.descriptor, image_pyramid, D, _tdata_pyr);
  parallel_for(Range(0, image_pyramid.size()), func);
#endif

}

Result VisualOdometryPose::estimatePose(const uint8_t* image_ptr, const Matrix44& T_init,
                                        Matrix44& T_est)
{
  Info("estimatePose\n");

  Result ret;
  T_est = T_init;
  auto& stats = ret.optimizerStatistics;
  stats.resize(_tdata_pyr.size());

  auto desc = MakeDescriptor(_params.descriptor);
  _image_pyramid.compute(ToOpenCV(image_ptr, _image_size));

  _pose_estimator.setParameters(_pose_est_params_low_res);
  for(int i = _image_pyramid.size()-1; i >= _params.maxTestLevel; --i)
  {
    Info("LEVEL %d\n", i);
    if(i >= _params.maxTestLevel)
    _pose_estimator.setParameters(_pose_est_params); // restore high res params

    if(_tdata_pyr[i]->numPixels() > _params.minNumPixelsToWork) {
      desc->compute(_image_pyramid[i]);
      stats[i] = _pose_estimator.run(_tdata_pyr[i].get(), desc.get(), T_est);
    } else {
      Warn("Not enough points at octave %d [needs %d]\n", i, _params.minNumPixelsToWork);
    }
  }

  ret.pose = T_est;
  return ret;
}

bool VisualOdometryPose::hasData() const
{
  return _tdata_pyr[_params.maxTestLevel]->numPoints() > 0;
}

} // bpvo

#undef WITH_THREAD_POOL
#undef WITH_TBB_TASK_GROUP

