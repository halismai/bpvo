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

#ifndef BPVO_UTILS_H
#define BPVO_UTILS_H

#include <bpvo/debug.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <string>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace std {
inline std::string to_string(std::string s) { return s; }
}; // std

namespace bpvo {

template <typename T> inline void
assert_is_floating_point()
{
  static_assert(std::is_floating_point<T>::value, "value must be floating point");
}

// TODO the arguments are ambigious! watch out
inline int sub2ind(const int stride, int r, int c)
{
  return r*stride + c;
}

inline void ind2sub(const int stride, const int ind, int& r, int& c)
{
  r = ind / stride;
  c = ind % stride;
}

template <int Alignment = 16, class T> inline
bool IsAligned(const T* ptr)
{
  return 0 == ((unsigned long) ptr & (Alignment-1));
}

/**
 * \return the minimum number >= size that is divisible by 'N'
 */
template <int N> static inline
size_t alignSize(size_t size)
{
  assert( !(N & (N-1)) );
  return (size + N-1) & -N;
}

/**
 * Align the pointer to N bytes
 */
template <typename T, int N = static_cast<int>(sizeof(T))> static inline
T* alignPtr(T* ptr)
{
  return (T*)(((size_t) ptr + N-1) & -N);
}

template <class ...T> inline void UNUSED(const T&...) {}

/**
 * round up the input argument n to be a multiple of m
 */
int roundUpTo(int n, int m);

/** vsprintf like */
std::string Format(const char* fmt, ...);

/**
 * return the date & time as a string
 */
std::string datetime();

/**
 * \return wall clock in seconds
 */
double GetWallClockInSeconds();

/**
 * \return the current unix timestamp
 */
uint64_t UnixTimestampSeconds();
inline uint64_t getTimestamp() { return UnixTimestampSeconds(); }

/**
 * \return the current unix timestamp as milliseconds
 */
uint64_t UnixTimestampMilliSeconds();

/**
 * sleep for the given milliseconds
 */
void Sleep(int32_t milliseconds);

/**
 * backtrace
 */
std::string GetBackTrace();

//
// string utilities
//

/** case insenstive string comparision */
bool icompare(const std::string& a, const std::string& b);

struct CaseInsenstiveComparator
{
  bool operator()(const std::string&, const std::string&) const;
}; // CaseInsenstiveComparator


/**
 * converts the input string to a number
 */
template <typename T> T str2num(const std::string&);

template <> int str2num<int>(const std::string& s);
template <> double str2num<double>(const std::string& s);
template <> float str2num<float>(const std::string& s);
template <> bool str2num<bool>(const std::string& s);

/**
 * converts string to number
 * \return false if conversion to the specified type 'T' fails
 *
 * e.g.:
 *
 * double num;
 * assert( true == str2num("1.6", num) );
 * assert( false == str2num("hello", num) );
 *
 */
template <typename T> inline
bool str2num(std::string str, T& num)
{
  std::istringstream ss(str);
  return !(ss >> num).bad();
}

/**
 * Uses the delimiter to split the string into tokens of numbers, e.g,
 *
 * string str = "1.2 1.3 1.4 1.5";
 * auto tokens = splitstr(str, ' ');
 *
 * // 'tokens' now has [1.2, 1.3, 1.4, 1.5]
 */
std::vector<std::string> splitstr(const std::string& str, char delim = ' ');

template <typename T> inline
std::vector<T> str2num(const std::vector<std::string>& strs)
{
  std::vector<T> ret(strs.size());
  for(size_t i = 0; i < strs.size(); ++i)
    ret[i] = str2num<T>(strs[i]);

  return ret;
}


template <typename T> inline
std::string tostring(const T& something)
{
  std::stringstream oss;
  oss << something;

  return oss.str();
}

double wallclock();

double cputime();

/** \return the date as a string */
std::string dateAsString();

/** \return time as a string */
std::string timeAsString();

std::string errno_string();

struct Error : public std::logic_error
{
  inline Error(std::string what)
      : logic_error(what) {}
}; // Error

#define THROW_ERROR(msg) \
    throw bpvo::Error(bpvo::Format("[ %s:%04d ] %s", MYFILE, __LINE__, msg))

#define THROW_ERROR_IF(cond, msg) if( !!(cond) ) THROW_ERROR( (msg) )

#define DIE_IF(cond, msg) if( !!(cond) ) Fatal((msg))

template <typename Iterator> static inline typename
Iterator::value_type median(Iterator first, Iterator last)
{
  auto n = std::distance(first, last);
  auto middle = first + n/2;
  std::nth_element(first, middle, last);
  //__gnu_parallel::nth_element(first, middle, last);

  if(n % 2 != 0) {
    return *middle;
  } else {
    auto m = std::max_element(first, middle);
    return (*m + *middle) / 2.0;
  }
}

template <class Container> static inline typename Container::
value_type median(Container& data)
{
  if(data.empty()) {
    Warn("median: empty data\n");
    return typename Container::value_type(0);
  }

  if(data.size() < 3)
    return data[0];

  return median(std::begin(data), std::end(data));
}

template <class Container> static inline typename Container::
value_type medianAbsoluteDeviation(Container& data)
{
  auto m = median(data);
  for(auto& v : data)
    v = std::abs(v - m);

  return median(data);
}

template <typename T, class Pred> inline
T clamp(const T& v, const T& min_val, const T& max_val, Pred pred)
{
  return pred(v, min_val) ? min_val : pred(max_val, v) ? max_val : v;
}

template <typename T> inline
T clamp(const T& v, const T& min_val, const T& max_val)
{
  return clamp(v, min_val, max_val, std::less<T>());
}


namespace fs {

/**
 * \return directory separator, this is a slash '/'
 */
std::string dirsep(std::string fn);

/**
 * Expands '~' to user's home directory
 */
std::string expand_tilde(std::string);


/**
 * \return the extension of the input filename
 */
std::string extension(std::string filename);

/**
 * \return true if path exists
 */
bool exists(std::string path);

/**
 * \return true if path is a regular file
 */
bool is_regular(std::string path);

/**
 * \return true if directory
 */
bool is_dir(std::string path);

/**
 * Creates a directory.
 *
 * \return name of the directory that was created (empty if we could not create
 * one for you)
 *
 * if 'try_unique' is true, the function will keep trying up to 'max_tries' to
 * create a unique directory
 */
std::string mkdir(std::string dname, bool try_unique = false, int max_tries = 1000);


}; // fs

}; // bpvo

#endif // BPVO_UTILS_H
