#include "utils/dataset_loader_thread.h"
#include "bpvo/utils.h"
#include "bpvo/debug.h"

namespace bpvo {

DatasetLoaderThread::DatasetLoaderThread(UniquePointer<Dataset> dataset,
                                         BufferType& buffer)
    : _dataset(std::move(dataset)), _buffer(buffer),
    _thread([=] { this->start(); }) { _is_running = true; }

DatasetLoaderThread::~DatasetLoaderThread() { stop(); }

bool DatasetLoaderThread::isRunning() const { return _is_running; }

void DatasetLoaderThread::stop(bool empty_buffer)
{
  _stop_requested = true;

  if(empty_buffer) {
    FramePointer frame;
    while(_buffer.pop(&frame, 10))
      ;
  }

  if(_thread.joinable())
    _thread.join();
}

void DatasetLoaderThread::start()
{
  FramePointer frame;
  int f_i = 0;

  try
  {
    while(nullptr != (frame = _dataset->getFrame(f_i++)) && !_stop_requested)
    {
      _buffer.push(std::move(frame));
      Sleep(2);
    }

  } catch(const std::exception& ex)
  {
    Warn("problem '%s'\n", ex.what());
    _buffer.push(nullptr);
  }

  _is_running = false;
}

} // bpvo

