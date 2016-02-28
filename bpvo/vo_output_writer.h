#ifndef BPVO_VO_OUTPUT_WRITER_H
#define BPVO_VO_OUTPUT_WRITER_H

#include <bpvo/vo_output.h>
#include <utils/bounded_buffer.h>

#include <thread>
#include <atomic>
#include <string>

namespace bpvo {

class VoOutputWriter
{
 public:
  typedef UniquePointer<VoOutput> VoOutputPointer;

  /** stores filename <-> pointer */
  typedef std::pair<std::string, VoOutputPointer> ElementType;

 public:
  inline VoOutputWriter(size_t capacity = 32)
      : _buffer(capacity), _thread(&VoOutputWriter::run, this) {}

  inline ~VoOutputWriter() { stop(); }

  inline void stop()
  {
    _stop = true;
    if(_thread.joinable())
      _thread.join();
  }

  inline bool isRunning() const { return !_stop; }

  inline void add(std::string output_fn, VoOutputPointer ptr)
  {
    _buffer.push({output_fn, std::move(ptr)});
  }

 protected:
  BoundedBuffer<ElementType> _buffer;
  std::thread _thread;
  std::atomic<bool> _stop{false};

  void run();
}; // VoOutputWriter

}; // bpvo

#endif // BPVO_VO_OUTPUT_WRITER_H
