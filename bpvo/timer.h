#ifndef BPVO_TIMER_H
#define BPVO_TIMER_H

#include <chrono>

namespace bpvo {
/**
 * Simple timer. The timer is enabled if BITPLANES_WITH_TIMING is enabled when
 * compiling code.
 */
class Timer
{
  typedef std::chrono::milliseconds Milliseconds;

 public:
  /**
   * By default the constructor start timing.
   */
  inline Timer() { start(); }

  /**
   * Calling start again will reset the time
   */
  void start();

  /**
   * Stops the timer. The start time point is reset
   */
  Milliseconds stop();

  /**
   * Keeps the timer running and reports elapsed milliseconds since the last
   * call to start() (or stop())
   */
  Milliseconds elapsed();

 protected:
  std::chrono::high_resolution_clock::time_point _start_time;
}; // Timer

/**
 * Times a piece of code by running N_rep and returns the average run time in
 * milliseconds
 *
 * Example
 *
 * MyClass my_class;
 * auto t = TimeCode(100, [=]() { my_class.runOperation(); });
 * std::cout << "Time: " << t << " ms\n";
 */
template <class Func, class ...Args> static inline
double TimeCode(int N_rep, Func&& f, Args... args)
{
  Timer timer;
  for(int i = 0; i < N_rep; ++i)
    f(args...);
  auto t = timer.stop();
  return t.count() / (double) N_rep;
}

}; // bpvo

#endif // BPVO_TIMER_H
