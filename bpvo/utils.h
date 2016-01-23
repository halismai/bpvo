#ifndef BPVO_UTILS_H
#define BPVO_UTILS_H

#include <cassert>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <string>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bpvo {

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
    throw Error(Format("[ %s:%04d ] %s", MYFILE, __LINE__, msg))

#define THROW_ERROR_IF(cond, msg) if( !!(cond) ) THROW_ERROR( (msg) )


}; // bpvo

#endif // BPVO_UTILS_H
