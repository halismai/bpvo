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

#ifndef BPVO_BOUNDED_BUFFER_H
#define BPVO_BOUNDED_BUFFER_H

#include <boost/circular_buffer.hpp>
#include <boost/call_traits.hpp>

#include <mutex>
#include <chrono>
#include <condition_variable>

namespace bpvo {

template <class T>
class BoundedBuffer
{
 public:
  typedef boost::circular_buffer<T> Container_t;
  typedef typename Container_t::size_type  size_type;
  typedef typename Container_t::value_type value_type;
  typedef typename boost::call_traits<value_type>::param_type param_type;

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
  bool pop(value_type* item, int wait_time_ms = 10);

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
  _container.push_front(item);
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
    *item = _container[--_unread];
    lock.unlock();
    _cond_not_full.notify_one();
    return true;
  } else {
    lock.unlock();
    return false;
  }
}

}; // bpvo

#endif // BPVO_BOUNDED_BUFFER_H
