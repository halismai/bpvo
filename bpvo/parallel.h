#ifndef BPVO_PARALLEL_H
#define BPVO_PARALLEL_H

namespace bpvo {

class Range
{
 public:
  inline Range() : _begin(0), _end(0) {}
  inline Range(int begin_, int end_) : _begin(begin_), _end(end_) {}

  inline int size() const { return _end - _begin; }
  inline bool empty() const { return _end == _begin; }

  inline int begin() const { return _begin; }
  inline int end() const { return _end; }

 protected:
  int _begin;
  int _end;
}; // Range

class ParallelForBody
{
 public:
  virtual ~ParallelForBody();
  virtual void operator()(const Range&) const = 0;
}; // ParallelForBody


int getNumThreads();
int getNumMaxThreads();
int getThreadNum();
int getNumberOfCPUs();
void setNumThreads(int);

void parallel_for(const Range&, const ParallelForBody&, double nstripes = -1);

static inline bool operator==(const Range& a, const Range& b)
{
  return a.begin() == b.begin() && a.end() == b.end();
}

static inline bool operator!=(const Range& a, const Range& b)
{
  return !(a == b);
}


}; // bpvo

#endif // BPVO_PARALLEL_H
