#ifndef BPVO_UTILS_DATASET_LOADER_THREAD_H
#define BPVO_UTILS_DATASET_LOADER_THREAD_H

#include <utils/dataset.h>
#include <utils/bounded_buffer.h>
#include <utils/stereo_calibration.h>

#include <atomic>
#include <thread>

namespace bpvo {

class DatasetLoaderThread
{
 public:
  typedef UniquePointer<DatasetFrame> FramePointer;
  typedef BoundedBuffer<FramePointer> BufferType;

 public:
  /**
   * \param pointer to the dataset loader
   * \param buffer shared buffer to store data
   */
  DatasetLoaderThread(UniquePointer<Dataset> dataset, BufferType& buffer);

  virtual ~DatasetLoaderThread();

  /**
   * stops the thread and optionally clears the buffer
   */
  void stop(bool empty_buffer = true);

  /**
   * \return true if the thread is still running
   */
  bool isRunning() const;

 protected:
  UniquePointer<Dataset> _dataset;
  BufferType& _buffer;

  std::atomic<bool> _stop_requested{false};
  std::atomic<bool> _is_running{false};

  std::thread _thread;

  void start();
}; // DatasetLoaderThread

}; // bpvo

#endif // BPVO_UTILS_DATASET_LOADER_THREAD_H

