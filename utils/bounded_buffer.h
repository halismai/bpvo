#ifndef BPVO_TEST_BOUNDED_BUFFER_H
#define BPVO_TEST_BOUNDED_BUFFER_H

#if !defined(WITH_BOOST)
#error "compile WITH_BOOST"
#endif

#include <boost/circular_buffer.hpp>
#include <boost/call_traits.hpp>

#include <mutex>
#include <chrono>
#include <condition_variable>

namespace bpvo {

//
// from boost examples
// http://www.boost.org/doc/libs/1_60_0/doc/html/circular_buffer/examples.html
//
template <class T>
class BoundedBuffer
{
 public:
  typedef boost::circular_buffer<T> Container_t;
  typedef typename Container_t::size_type  size_type;
  typedef typename Container_t::value_type value_type;
  // call_traits is useless in this department
  //typedef typename boost::call_traits<value_type>::param_type param_type;
  typedef value_type&& param_type;

 public:
  /**
   * Initializers a buffers with at most 'capacity' elements
   */
  explicit BoundedBuffer(size_type capacity);

  /**
   * pushes an element into the buffer
   * If the buffer is full, the function waits until a slot is available
   */
  void push(param_type item);

  /**
   * set 'item' to the first element that was pushed into the buffer.
   *
   * The function waits for the specified milliseconds for data if the buffer
   * was empty
   *
   * \return true if we popped something, false otherwise (timer has gone off)
   */
  bool pop(value_type* item, int wait_time_ms = 1);

  /**
   * \return true if the buffer is full (if possible)
   */
  bool full();

  /**
   * \return the size of the buffer (number of elements) if we are able to get a
   * lock. If not, we return -1
   */
  int size();

 private:
  BoundedBuffer(const BoundedBuffer&) = delete;
  BoundedBuffer& operator=(const BoundedBuffer&) = delete;

 private:
  size_type _unread;
  Container_t _container;
  std::mutex _mutex;
  std::condition_variable _cond_not_empty;
  std::condition_variable _cond_not_full;
}; // BoundedBuffer

template <typename T> inline
BoundedBuffer<T>::BoundedBuffer(size_type capacity) :
    _unread(0), _container(capacity) {}

template <typename T> inline
void BoundedBuffer<T>::push(param_type item)
{
  std::unique_lock<std::mutex> lock(_mutex);
  _cond_not_full.wait(lock, [=] { return _unread < _container.capacity(); });
  _container.push_front(std::move(item));
  ++_unread;
  lock.unlock();
  _cond_not_empty.notify_one();
}

template <typename T> inline
bool BoundedBuffer<T>::pop(value_type* item, int wait_time_ms)
{
  std::unique_lock<std::mutex> lock(_mutex);
  if(_cond_not_empty.wait_for(lock, std::chrono::milliseconds(wait_time_ms),
                              [=] { return _unread > 0; } )) {
    (*item).swap(_container[--_unread]);
    lock.unlock();
    _cond_not_full.notify_one();
    return true;
  } else {
    lock.unlock();
    return false;
  }
}

template <typename T> inline
bool BoundedBuffer<T>::full()
{
  if(_mutex.try_lock()) {
    bool ret = _container.full();
    _mutex.unlock();
    return ret;
  }

  return false;
}

template <typename T> inline
int BoundedBuffer<T>::size()
{
  int ret = -1;
  if(_mutex.try_lock()) {
    ret = static_cast<int>(_container.size());
    _mutex.unlock();
  }

  return ret;
}


}; // bpvo

#endif // DIVO_UTIL_BOUNDED_BUFFER_H
