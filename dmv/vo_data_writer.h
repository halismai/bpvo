#ifndef BPVO_DMV_VO_DATA_WRITER_H
#define BPVO_DMV_VO_DATA_WRITER_H

#include <dmv/vo_data.h>
#include <utils/bounded_buffer.h>

#include <utility>
#include <atomic>
#include <thread>

namespace bpvo {
namespace dmv {

class VoDataWriterThread
{
 public:
  typedef UniquePointer<VoDataAbstract> VoDataPtr;
  typedef std::pair<std::string, VoDataPtr> ElementType;
  typedef BoundedBuffer<ElementType> BufferType;

 public:
  VoDataWriterThread(size_t capacity = 32) :
      _buffer(capacity), _thread(&VoDataWriterThread::run, this) {}

  ~VoDataWriterThread() { stop(); }

  void stop();

 protected:
  void run();

  std::atomic<bool> _stop{false};
}; // VoDataWriterThread


}; // dmv
}; // bpvo

#endif // BPVO_DMV_VO_DATA_WRITER_H
