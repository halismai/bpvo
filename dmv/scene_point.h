#ifndef BPVO_DMV_SCENE_POINT_H
#define BPVO_DMV_SCENE_POINT_H

#include <bpvo/types.h>
#include <bpvo/utils.h>

#include <algorithm>
#include <vector>

namespace bpvo {
namespace dmv {

template <class Descriptor>
class ScenePoint
{
 public:
  typedef uint16_t IdType;
  typedef std::vector<IdType> IdList;

 public:
  inline ScenePoint(IdList frame_id, const Point& X, const Descriptor& d)
      : _X(X), _desc(d)
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

  inline bool hasFrameId(IdType id) const {
    return std::find(_frame_ids.begin(), _frame_ids.end(), id) != _frame_ids.end();
  }

  inline void addFrameId(IdType id) const {
    THROW_ERROR_IF(hasFrameId(id),
                 Format("frame id %d exists in frame id list", id).c_str());

    _frame_ids.push_back(id);
  }

 public:

 private:
  Point _X;
  Descriptor _desc;
  IdList _frame_ids;

}; // ScenePoint

}; // dmv
}; // bpvo

#endif // BPVO_DMV_SCENE_POINT_H
