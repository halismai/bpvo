#include <bpvo/debug.h>

#if !defined(WITH_CERES)
int main() { Fatal("compile WITH_CERES\n"); }
#else

#include <bpvo/vo_output.h>
#include <bpvo/vo_output_reader.h>
#include <bpvo/eigen.h>
#include <dmv/photometric_vo.h>

#include <fstream>
#include <iostream>

using namespace bpvo;

template <class Point3Vector> inline
void writePoints(std::string fn, const Point3Vector& pts)
{
  std::ofstream ofs(fn);
  if(ofs.is_open())
  {
    for(size_t i = 0; i < pts.size(); ++i)
      ofs << pts[i].transpose() << "\n";
  }
}

int main()
{
  Mat_<double,3,3> K;
  K <<
      615.0, 0.0, 320.0,
      0.0, 615.0, 240.0,
      0.0, 0.0, 1.0;

  dmv::PhotometricVoConfig config;

  config.withSpatialWeighting = false;
  config.intensityScale = 1.0;

  dmv::PhotometricVo vo(K, 0.1, config);

  VoOutputReader vo_output_reader(".", "vo_%05d.voout", 0);

  auto frame = vo_output_reader[0];

  std::cout << "initial pose\n" << frame->pose() << std::endl;
  vo.addFrame(frame.get());
  auto result = vo.addFrame(frame.get());

  std::cout << "optimized\n" << result.pose << std::endl;


  writePoints("X0.txt", result.pointsRaw);
  writePoints("X1.txt", result.points);

  return 0;

}

#endif
