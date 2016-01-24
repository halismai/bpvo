#include "bpvo/timer.h"

namespace bpvo {

#define BITPLANES_WITH_TIMING 1

void Timer::start()
{
#if BITPLANES_WITH_TIMING
  _start_time = std::chrono::high_resolution_clock::now();
#endif
}

auto Timer::stop() -> Milliseconds
{
#if BITPLANES_WITH_TIMING
  auto t_now = std::chrono::high_resolution_clock::now();
  auto ret = std::chrono::duration_cast<Milliseconds>(t_now - _start_time);
  _start_time = t_now;
  return ret;
#else
  return Milliseconds();
#endif
}

auto Timer::elapsed() -> Milliseconds
{
#if BITPLANES_WITH_TIMING
  auto t_now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<Milliseconds>(t_now - _start_time);
#else
  return Milliseconds();
#endif
}


}
