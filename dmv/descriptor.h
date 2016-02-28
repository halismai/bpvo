#ifndef BPVO_DMV_DESCRIPTOR_H
#define BPVO_DMV_DESCRIPTOR_H

#include <bpvo/types.h>

namespace bpvo {
namespace dmv {

class Patch3x3;
namespace detail {

template <class> struct descriptor_traits;

template <> struct descriptor_traits<Patch3x3>
{
  static constexpr std::size_t Dimension = 9;
  typedef double  DataType;
}; // Patch3x3

}; // detail

template <class Derived>
class DescriptorBase
{
 public:
  typedef detail::descriptor_traits<Derived> traits;

  static constexpr std::size_t Dimension = traits::Dimension;
  typedef typename traits::DataType   DataType;

 public:
  inline std::size_t size() const { return Dimension; }

  inline const DataType* data() const { return derived()->data(); }

  template <class ImageT> inline
  void setFromImage(const ImageT& I, const ImagePoint& p)
  {
    derived()->set(I, p);
  }

 protected:

  inline const Derived* derived() const { return static_cast<const Derived*>(this); }
  inline       Derived* derived()      { return static_cast<Derived*>(this); }
}; // DescriptorBase

}; // dmv
}; // bpvo

#endif // BPVO_DMV_DESCRIPTOR_H
