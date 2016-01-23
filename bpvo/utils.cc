#include <unistd.h> // have this included first
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <execinfo.h>
#include <errno.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <thread>
#include <cstdarg>
#include <functional>

#include "bpvo/utils.h"
#include "bpvo/debug.h"

namespace bpvo {

bool icompare(const std::string& a, const std::string& b)
{
  return !strncasecmp(a.c_str(), b.c_str(), a.size());
}

bool CaseInsenstiveComparator::operator()(const std::string& a,
                                          const std::string& b) const
{
  return strncasecmp(a.c_str(), b.c_str(), std::min(a.size(), b.size())) < 0;
}

template<> int str2num<int>(const std::string& s) { return std::stoi(s); }

template <> double str2num<double>(const std::string& s) { return std::stod(s); }

template <> float str2num<float>(const std::string& s) { return std::stof(s); }

template <> bool str2num<bool>(const std::string& s)
{
  if(icompare(s, "true")) {
    return true;
  } else if(icompare(s, "false")) {
    return false;
  } else {
    // try to parse a bool from int {0,1}
    int v = str2num<int>(s);
    if(v == 0)
      return false;
    else if(v == 1)
      return true;
    else
      throw std::invalid_argument("string is not a boolean");
  }
}

int roundUpTo(int n, int m)
{
  return m ? ( (n % m) ? n + m - (n % m) : n) : n;
}

using std::string;
using std::vector;

vector<string> splitstr(const std::string& str, char delim)
{
  std::vector<std::string> ret;
  std::stringstream ss(str);
  std::string token;
  while(std::getline(ss, token, delim))
    ret.push_back(token);

  return ret;
}

string datetime()
{
  auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

  char buf[128];
  std::strftime(buf, sizeof(buf), "%a %b %d %H:%M:%S %Z %Y", std::localtime(&tt));
  return std::string(buf);
}


double GetWallClockInSeconds()
{
  return wallclock();
}

uint64_t UnixTimestampSeconds()
{
  return std::chrono::seconds(std::time(NULL)).count();
}

uint64_t UnixTimestampMilliSeconds()
{
  return static_cast<int>( 1000.0 * UnixTimestampSeconds() );
}

void Sleep(int32_t ms)
{
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

string Format(const char* fmt, ...)
{
  vector<char> buf(1024);

  while(true) {
    va_list va;
    va_start(va, fmt);
    auto len = vsnprintf(buf.data(), buf.size(), fmt, va);
    va_end(va);

    if(len < 0 || len >= (int) buf.size()) {
      buf.resize(std::max((int)(buf.size() << 1), len + 1));
      continue;
    }

    return string(buf.data(), len);
  }
}

string GetBackTrace()
{
  void* buf[1024];
  int n = backtrace(buf, 1024);
  char** strings = backtrace_symbols(buf, n);
  if(!strings) {
    perror("backtrace_symbols\n");
    return "";
  }

  std::string ret;
  for(int i = 0; i < n; ++i)
    ret += (std::string(strings[i]) + "\n");

  free(strings);

  return ret;
}

string dateAsString()
{
  std::time_t t;
  std::time(&t);

  char buf[64];
  std::strftime(buf, 64, "%Y-%m-%d", std::localtime(&t));

  return std::string( buf );
}

string timeAsString()
{
  std::time_t t;
  std::time(&t);

  char buf[64];
  std::strftime(buf, 64, "%A, %d.%B %Y, %H:%M", std::localtime(&t));

  return std::string( buf );
}

double wallclock()
{
  struct timeval tv;
  if( -1 == gettimeofday( &tv, NULL ) ) {
    Warn("could not gettimeofday, error: '%s'\n", errno_string().c_str());
  }

  return static_cast<double>( tv.tv_sec )
      + static_cast<double>( tv.tv_usec ) / 1.0E6;
}

double cputime()
{
  struct rusage ru;
  if( -1 == getrusage( RUSAGE_SELF, &ru ) ) {
    Warn("could not cpu usage, error '%s'\n", errno_string().c_str());
  }

  return static_cast<double>( ru.ru_utime.tv_sec )
      + static_cast<double>( ru.ru_utime.tv_usec ) / 1.0E6;

}


}; // bpvo


