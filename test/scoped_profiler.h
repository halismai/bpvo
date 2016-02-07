#ifndef BPVO_TEST_SCOPED_PROFILER
#define BPVO_TEST_SCOPED_PROFILER

#include <string>

namespace bpvo {

class ScopedProfiler
{
 public:
  ScopedProfiler(std::string name);
  ~ScopedProfiler();

  void stop();

 private:
  bool _is_running = false;
}; // ScopedProfiler

}; // bpvo

#endif // BPVO_TEST_SCOPED_PROFILER
