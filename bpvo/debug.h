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

#ifndef BPVO_DEBUG_H
#define BPVO_DEBUG_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#define FORCE_INLINE inline __attribute__((always_inline))
#define NO_INLINE           __attribute__((noinline))
#define ALIGNED(...)        __attribute__((aligned(__VA_ARGS__)))

#define likely(expr)        __builtin_expect((expr),true)
#define unlikey(expr)       __builtin_expect((expr),false)

#define ANSI_COLOR_BLACK            0
#define ANSI_COLOR_RED              1
#define ANSI_COLOR_GREEN            2
#define ANSI_COLOR_YELLOW           3
#define ANSI_COLOR_BLUE             4
#define ANSI_COLOR_MAGENTA          5
#define ANSI_COLOR_CYAN             6
#define ANSI_COLOR_WHITE            7
#define ANSI_COLOR_DEFAULT          9

#define ANSI_FG                     30
#define ANSI_BG                     40

#define ANSI_FORMAT_BOLD_ON         1
#define ANSI_FORMAT_ITALIC_ON       3
#define ANSI_FORMAT_UNDERLINE_ON    4
#define ANSI_FORMAT_INVERSE_ON      7
#define ANSI_FORMAT_STRIKE_ON       9

#define ANSI_FORMAT_BOLD_OFF        22
#define ANSI_FORMAT_ITALIC_OFF      23
#define ANSI_FORMAT_UNDERLINE_OFF   24
#define ANSI_FORMAT_INVERSE_OFF     27
#define ANSI_FORMAT_STRIKE_OFF      29

#define ANSI_SET(file, val) fprintf((file), "%c[%dm", 27, (val))

#define MYFILE (strrchr(__FILE__,'/') ? strrchr(__FILE__,'/')+1:__FILE__)

#define WHR_STR "[ %s:%04d ]: "
#define WHR_ARG MYFILE,__LINE__

#ifndef NDEBUG
#define DPRINT(...) fprintf(stdout, __VA_ARGS__)
#ifndef NO_TTY_COLOR
#define dprintf(args...) do {                     \
  ANSI_SET(stdout,ANSI_COLOR_BLUE+ANSI_FG);       \
  fprintf(stdout, WHR_STR, WHR_ARG);              \
  ANSI_SET(stdout,0);                             \
  fprintf(stdout, args);                          \
} while(0)
#else
#define dprintf(args...) do {                     \
  fprintf(stdout, WHR_STR, WHR_ARG);              \
  fprintf(stdout, args);                          \
} while(0)
#endif
#else // do nothing
#define dprintf(_fmt, ...) {}
#endif // NDEBUG


#ifndef NO_TTY_COLOR

#define Fatal(args...) do {                       \
  ANSI_SET(stderr, ANSI_COLOR_RED+ANSI_FG);       \
  fprintf(stderr, WHR_STR, WHR_ARG);              \
  fprintf(stderr, args);                          \
  ANSI_SET(stderr,0);                             \
  exit(1);                                        \
} while (0)

#define Warn(args...) do {                        \
  ANSI_SET(stderr, ANSI_COLOR_YELLOW+ANSI_FG);    \
  fprintf(stderr, WHR_STR, WHR_ARG);              \
  fprintf(stderr, args);                          \
  ANSI_SET(stderr,0);                             \
} while (0)

#define Info(args...) do {                        \
  ANSI_SET(stdout, ANSI_COLOR_GREEN+ANSI_FG);     \
  fprintf(stdout, WHR_STR, WHR_ARG);              \
  fprintf(stdout, args);                          \
  ANSI_SET(stderr,0);                             \
} while (0)

#define MissingLibraryError(name) do {            \
  Fatal(name " is requried to run this app\n");   \
} while(0)


#else  // do not use colors

#define Fatal(args...) do {                       \
  fprintf(stderr, WHR_STR, WHR_ARG);              \
  fprintf(stderr, args);                          \
  exit(1);                                        \
} while(0)

#define Warn(args...) do {                        \
  fprintf(stderr, WHR_STR, WHR_ARG);              \
  fprintf(stderr, args);                          \
} while(0)

#define Info(args...) do {                        \
  fprintf(stdout, WHR_STR, WHR_ARG);              \
  fprintf(stdout, args);                          \
} while(0)

#define MissingLibraryError(name) do {            \
  Fatal(name " is requried to run this app");     \
} while(0)

#endif // NO_TTY_COLOR


#endif // BPVO_DEBUG_H
