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

#ifndef BPVO_ALIGNED_ALLOCATOR_H
#define BPVO_ALIGNED_ALLOCATOR_H

#include <memory>

template <typename T, std::size_t Alignment = 16>
class AlignedAllocator : public std::allocator<T>
{
 public:
  typedef typename std::allocator<T>::size_type     size_type;
  typedef typename std::allocator<T>::pointer       pointer;
  typedef typename std::allocator<T>::const_pointer const_pointer;

 public:
  template <typename U>
  struct rebind { typedef AlignedAllocator<U,Alignment> other; };

 public:
  AlignedAllocator() {}
  AlignedAllocator(const AlignedAllocator& o) : std::allocator<T>(o) {}
  ~AlignedAllocator() {}

  template <typename U, std::size_t A>
  bool operator==(const AlignedAllocator<U,A>&) const { return true; }

  template <typename U, std::size_t A>
  bool operator!=(const AlignedAllocator<U,A>&) const { return false; }

  inline pointer allocate(size_type n, const_pointer = nullptr)
  {
    // TODO some systems might not have posix_memalign
    void* ret = nullptr;
    if(posix_memalign(&ret, Alignment, n*sizeof(T)))
      throw std::bad_alloc();

    return static_cast<pointer>(ret);
  }

  inline void deallocate(pointer ptr, size_type){ std::free(ptr); }

}; // AlignedAllocator

#endif // BPVO_ALIGNED_ALLOCATOR_H
