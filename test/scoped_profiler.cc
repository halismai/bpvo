#include <bpvo/utils.h>
#include "test/scoped_profiler.h"

#if defined(WITH_PROFILER)
#include <gperftools/profiler.h>
#endif

namespace bpvo {


ScopedProfiler::ScopedProfiler(std::string name)
    : _is_running(true)
{
  UNUSED(name);

#if defined(WITH_PROFILER)
  ProfilerStart(name.c_str());
#endif
}

ScopedProfiler::~ScopedProfiler()
{
  stop();
}

void ScopedProfiler::stop()
{
#if defined(WITH_PROFILER)
  if(_is_running) {
    //ProfilerFlush();
    ProfilerStop();
    _is_running = false;
  }
#endif
}

}; // bpvo
