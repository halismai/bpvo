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

#include <unistd.h> // have this included first
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <execinfo.h>
#include <errno.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>
#include <cstdarg>
#include <functional>

#include "bpvo/utils.h"
#include "bpvo/debug.h"

namespace bpvo {

bool icompare(const std::string& a, const std::string& b)
{
  return a.size() == b.size() ? !strncasecmp(a.c_str(), b.c_str(), a.size()) : false;
}

struct NoCaseCmp {
  inline bool operator()(const unsigned char& c1,
                         const unsigned char& c2) const
  {
    return std::tolower(c1) < std::tolower(c2);
  }
}; // NoCaseCmp

bool CaseInsenstiveComparator::operator()(const std::string& a,
                                          const std::string& b) const
{
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
                                      NoCaseCmp());
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

string errno_string()
{
  char buf[128];
  strerror_r(errno, buf, 128);
  return string(buf);
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

namespace fs {
string expand_tilde(string fn)
{
  if(fn.front() == '~') {
    string home = getenv("HOME");
    if(home.empty()) {
      std::cerr << "could not query $HOME\n";
      return fn;
    }

    // handle the case when name == '~' only
    return home + dirsep(home) + ((fn.length()==1) ? "" :
                                  fn.substr(1,std::string::npos));
  } else {
    return fn;
  }
}

string dirsep(string dname)
{
  return (dname.back() == '/') ? "" : "/";
}

string extension(string filename)
{
  auto i = filename.find_last_of(".");
  return (string::npos != i) ? filename.substr(i) : "";
}

bool exists(string path)
{
  struct stat buf;
  return (0 == stat(path.c_str(), &buf));
}

bool is_regular(string path)
{
  struct stat buf;
  return (0 == stat(path.c_str(), &buf)) ? S_ISREG(buf.st_mode) : false;
}

bool is_dir(string path)
{
  struct stat buf;
  return (0 == stat(path.c_str(), &buf)) ? S_ISDIR(buf.st_mode) : false;
}

bool try_make_dir(string dname, int mode = 0777)
{
  return (0 == ::mkdir(dname.c_str(), mode));
}

string mkdir(string dname, bool try_unique, int max_tries)
{
  if(!try_unique) {
    return try_make_dir(dname.c_str()) ? dname : "";
  } else {
    auto buf_len = dname.size() + 64;
    char* buf = new char[buf_len];
    int n = 0;
    snprintf(buf, buf_len, "%s-%05d", dname.c_str(), n);

    string ret;
    while(++n < max_tries) {
      if(try_make_dir(string(buf))) {
        ret = string(buf);
        break;
      }
    }

    delete[] buf;
    return ret;
  }
}

}; // fs

}; // bpvo


