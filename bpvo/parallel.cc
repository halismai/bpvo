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

#include "bpvo/parallel.h"

// based on opencv's parallel_for below is their notice

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if defined(WITH_TBB)
#include <tbb/tbb_stddef.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task.h>
#include <tbb/tbb.h>
#elif defined(WITH_OPENMP)
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <thread>

namespace bpvo {

ParallelForBody::~ParallelForBody() {}

static int s_numThreads = -1;

#if defined(WITH_TBB)
static tbb::task_scheduler_init s_taskScheduler(tbb::task_scheduler_init::deferred);
//static tbb::task_scheduler_init s_taskScheduler(tbb::task_scheduler_init::automatic);
#endif

int getNumThreads()
{
  if(s_numThreads == 0)
    return 1;

#if defined(WITH_TBB)
  return s_taskScheduler.is_active() ? s_numThreads : tbb::task_scheduler_init::default_num_threads();
#elif defined(WITH_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

int getNumMaxThreads()
{
  return getNumberOfCPUs();
}

void setNumThreads(int n)
{
  s_numThreads = n;

#if defined(WITH_TBB)
  if(s_taskScheduler.is_active())
    s_taskScheduler.terminate();
  if(n > 0)
    s_taskScheduler.initialize(n);
#elif defined(WITH_OPENMP)
  if(!omp_in_parallel()) {
    omp_set_num_threads( n > 0 ? n : s_numThreads );
  }
#endif
}

int getThreadNum()
{
#if defined(WITH_TBB)

#if TBB_INTERFACE_VERSION >= 6100 && defined(TBB_PREVIEW_TASK_ARENA) && TBB_PREVIEW_TASK_ARENA
  return tbb::task_arena::current_slot();
#else
  return 0;
#endif

#elif defined(WITH_OPENMP)
  return omp_get_thread_num();
#else
  return 0;
#endif
}

int getNumberOfCPUs()
{
  return std::thread::hardware_concurrency();
}


namespace {

class ParallelForWrapper
{
 public:
  ParallelForWrapper(const ParallelForBody& body, const Range& r, double nstripes)
      : _body(&body), _range(r)
  {
    double len = static_cast<double>( _range.size() );
    _nstripes = std::round(nstripes <= 0 ? len : std::min(std::max(nstripes, 1.0), len));
  }

  inline void operator()(const Range& sr) const
  {
    int len = _range.size(),
        begin = static_cast<int>(
            _range.begin() + (sr.begin()*len + _nstripes/2)/_nstripes),
        end = sr.end() >= _nstripes ? _range.end() :
            static_cast<int>(_range.begin() +
                (sr.end()*len + _nstripes/2) / _nstripes);

    Range r(begin, end);
    (*_body)(r);
  }

  inline Range stripeRange() const { return Range(0, _nstripes); }

 protected:
  const ParallelForBody* _body;
  Range _range;
  int _nstripes;
}; // ParallelForWrapper

#if defined(WITH_TBB)
class ParallelLoopProxy : public ParallelForWrapper
{
 public:
  ParallelLoopProxy(const ParallelForBody& body, const Range& r, double nstripes)
      : ParallelForWrapper(body, r, nstripes) {}

  inline void operator()(const tbb::blocked_range<int>& range) const
  {
    this->ParallelForWrapper::operator()(Range(range.begin(), range.end()));
  }
}; // ParallelLoopProxy
#else
typedef ParallelForWrapper ParallelLoopProxy;
#endif

} // namespace

void parallel_for(const Range& range, const ParallelForBody& body, double nstripes)
{
  if(s_numThreads != 0)
  {
    ParallelLoopProxy pbody(body, range, nstripes);
    Range srange(pbody.stripeRange());

    if(srange.size() == 1) {
      body(range);
      return;
    }

#if defined(WITH_TBB)
    tbb::parallel_for(tbb::blocked_range<int>(srange.begin(), srange.end()), pbody);
#else
#if defined(WITH_OPENMP)
#pragma omp parallel for schedule(dynamic)
#else
#endif
    for(int i = srange.begin(); i < srange.end(); ++i)
      pbody(Range(i, i + 1));
#endif
  } else
  {
    body(range);
  }
}


} // bpvo
