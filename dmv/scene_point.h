#ifndef BPVO_DMV_SCENE_POINT_H
#define BPVO_DMV_SCENE_POINT_H

#include <bpvo/types.h>
#include <bpvo/utils.h>

#include <algorithm>
#include <vector>

#include <dmv/zncc_patch.h>
#include <iostream>

namespace bpvo {
namespace dmv {

template <class Derived> static inline
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime - 1, 1>
normHomog(const Eigen::MatrixBase<Derived>& p)
{
  static_assert( Derived::RowsAtCompileTime != Eigen::Dynamic,
                "matrix size must be known at compile time" );

  auto w_i = typename Derived::Scalar(1.0) / p[Derived::RowsAtCompileTime-1];
  return Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime-1,1>(
      w_i * p.template head<Derived::RowsAtCompileTime-1>());
}

template <class Descriptor>
class ScenePoint
{
 public:
  typedef uint16_t IdType;
  typedef std::vector<IdType> IdList;
  typedef ZnccPatch<2,float> ZnccPatchType;

 public:
  inline ScenePoint(IdType frame_id, const Point& X, const Descriptor& d,
                    const ZnccPatchType& p)
      : _id(GetUniqueId()), _X(X), _desc(d), _zncc_patch(p)
  {
    _frame_ids.reserve(8);
    _frame_ids.push_back(frame_id);
  }

  inline const Point& X() const { return _X; }
  inline       Point& X()       { return _X; }

  inline void setDepth(double z) { _X.z() = z; }
  inline void setDepth(float z) { _X.z() = z; }

  inline const Descriptor& desc() const { return _desc; }
  inline       Descriptor& desc()       { return _desc; }

  inline const IdList& frameIds() const { return _frame_ids; }
  inline size_t numFrames() const { return _frame_ids.size(); }

  inline IdType referenceFrameId() const { return _frame_ids.front(); }

  inline bool hasFrameId(IdType id) const
  {
    return std::find(_frame_ids.begin(), _frame_ids.end(), id) != _frame_ids.end();
  }

  inline void addFrameId(IdType id)
  {
    THROW_ERROR_IF(hasFrameId(id),
                 Format("frame id %d exists in frame id list", id).c_str());

    _frame_ids.push_back(id);
  }

  inline void setZnccPatch(const ZnccPatchType& p) { _zncc_patch = p; }

  inline const ZnccPatchType& znccPatch() const { return _zncc_patch; }

  inline ImagePoint project(const Matrix34& P) const
  {
    return normHomog( P * _X );
  }

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

 private:
  uint64_t _id;
  Point _X;
  Descriptor _desc;
  IdList _frame_ids;

  ZnccPatchType _zncc_patch; // 5x5 patch for visibility checks

  friend std::ostream& operator<<(std::ostream& os, const ScenePoint& p)
  {
    os << p._id << ": " << (p._X.template head<3>()).transpose() << "\n";
    os << "frames = [";
    for(size_t i = 0; i < p._frame_ids.size(); ++i)
      os << p._frame_ids[i] << ((i != p._frame_ids.size()-1) ? "," : "");
    os << "]";

    return os;
  }

}; // ScenePoint

}; // dmv
}; // bpvo

#endif // BPVO_DMV_SCENE_POINT_H
