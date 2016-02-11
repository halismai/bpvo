#include "bpvo/point_cloud.h"

using namespace bpvo;

int main()
{
  static const int N = 1000;
  PointWithInfoVector v(N);
  for(int i = 0; i < N; ++i) {
    v[i].xyzw().setRandom(); v[i].xyzw()[3] = 1.0f;
    v[i].rgba().head<3>().setRandom();
    v[i].rgba()[3] = 255;
  }

  ToPlyFile("test.ply", v);

  return 0;
}
