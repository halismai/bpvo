#ifndef BPVO_DMV_TRAJECTORY_ID_H
#define BPVO_DMV_TRAJECTORY_ID_H

#include <bpvo/types.h>
#include <bpvo/utils.h>

#include <vector>
#include <algorithm>

namespace bpvo {
namespace dmv {

struct PoseTransformIdentity
{
 protected:
  static Matrix44 Apply(const Matrix44& pose) { return pose; }

  static Point3 CameraCenter(const Matrix44& pose)
  {
    return (pose.inverse()).block<3,1>(0,0);
  }
}; // PoseTransformIdentity

struct PoseTransformInverse
{
 protected:
  static Matrix44 Apply(const Matrix44& pose) { return pose.inverse(); }

  static Point3 CameraCenter(const Matrix44& pose) { return pose.block<3,1>(0,0); }
}; // PoseTransformInverse

template <class PoseTransform = PoseTransformInverse>
class TrajectoryWithId : private PoseTransform
{
 public:
  inline TrajectoryWithId() {}

  inline TrajectoryWithId(const Matrix44& pose, int id)
  {
    push_back(pose, id);
  }

  inline void push_back(const Matrix44& pose, int id)
  {
    THROW_ERROR_IF( !isIdUnique(id), Format("pose with id %d exists", id).c_str());

    if(!_data.empty())
      _data.push_back( PoseWithId(back() * PoseTransform::Apply(pose), id) );
    else
      _data.push_back( PoseWithId(PoseTransform::Apply(pose), id) );
  }

  inline const Matrix44& back() const { return _data.back().pose; }

  inline const Matrix44& operator[](size_t i) const
  {
    assert( i <  _data.size() );
    return _data[i].pose;
  }

  inline Matrix44& operator[](size_t i)
  {
    assert( i <  _data.size() );
    return _data[i].pose;
  }

  inline const Matrix44& atId(const int id) const
  {
    auto it = find_pose_with_id(id);
    THROW_ERROR_IF( it == _data.end(), Format("could not find pose with id %d\n", id) );
    return it->pose;
  }

  inline Matrix44& atId(const int id)
  {
    auto it = find_pose_with_id(id);
    THROW_ERROR_IF( it == _data.end(), Format("could not find pose with id %d\n", id) );
    return it->pose;
  }

  inline std::vector<Matrix44> getPoses() const
  {
    std::vector<Matrix44> ret(size());
    for(size_t i = 0; i < size(); ++i)
      ret[i] = _data[i].pose;

    return ret;
  }

  inline std::vector<Point3> getCameraCenters() const
  {
    std::vector<Point3> ret(size());
    for(size_t i = 0; i < size(); ++i)
      ret[i] = PoseTransform::CameraCenter(_data[i].pose);

    return ret;
  }

  inline bool isIdUnique(int id) const
  {
    return std::end(_data) == find_pose_with_id(id);
  }

  inline size_t size() const { return _data.size(); }

 private:
  struct PoseWithId
  {
    inline PoseWithId() {}

    inline PoseWithId(const Matrix44& pose_, int id_)
        : pose(pose_), id(id_) {}

    Matrix44 pose;
    int id;

   private:
    uint8_t _pad[12];

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  }; // PoseWithId

  std::vector<PoseWithId> _data;

  typedef typename std::vector<PoseWithId>::iterator Iterator;
  typedef typename std::vector<PoseWithId>::const_iterator ConstIterator;

  inline Iterator find_pose_with_id(int id)
  {
    for(auto it = _data.begin(); it != _data.end(); ++it)
      if(it->id == id)
        return it;

    return _data.end();
  }

  inline ConstIterator find_pose_with_id(int id) const
  {
    for(auto it = _data.begin(); it != _data.end(); ++it)
      if(it->id == id)
        return it;

    return _data.end();
  }

}; // TrajectoryWithId

}; // dmv
}; // bpvo

#endif // BPVO_DMV_TRAJECTORY_ID_H

