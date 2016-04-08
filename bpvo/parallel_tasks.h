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

#ifndef BPVO_PARALLEL_TASKS_H
#define BPVO_PARALLEL_TASKS_H

#if defined(WITH_TBB)
#include <tbb/task_group.h>
#else
#include <externals/ThreadPool.h>
#include <vector>
#include <future>
#endif

#include <bpvo/types.h>
#include <bpvo/utils.h>

namespace bpvo {

class ParallelTasks
{
 public:
  /**
   */
  inline ParallelTasks(int n_tasks)
  {
#if defined(WITH_TBB)
    _pool = make_unique<tbb::task_group>();
    UNUSED(n_tasks);
#else
    _pool = make_unique<ThreadPool>(n_tasks);
#endif
  }

  /**
   */
  template <class F> inline
  void add(const F& f)
  {
#if defined(WITH_TBB)
    _pool->run(f);
#else
    _results.emplace_back( _pool->enqueue(f) );
#endif
  }

  /**
   */
  void wait()
  {
#if defined(WITH_TBB)
    _pool->wait();
#else
    for(auto&& r : _results)
      r.get();
#endif
  }

 private:
#if defined(WITH_TBB)
  UniquePointer<tbb::task_group> _pool;
#else
  UniquePointer<ThreadPool> _pool;
  std::vector< std::future<void> > _results;
#endif
}; // ParallelTasks

}; // bpvo

#endif // BPVO_PARALLEL_TASKS_H

