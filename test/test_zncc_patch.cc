#include <dmv/zncc_patch.h>
#include <vector>

#include <bpvo/timer.h>

inline float Rand01()
{
  return rand() / static_cast<float>(RAND_MAX);
}

int main()
{
  int rows = 480, cols = 640;
  cv::Mat I(rows, cols, CV_8UC1);
  cv::randu(I, 0, 255);

  int N = 100 * 100;
  typedef bpvo::dmv::ZnccPatch<5,float> PatchType;
  std::vector<PatchType> patches(N);

  for(int i = 0; i < N; ++i)
  {
    bpvo::ImagePoint p(PatchType::Radius + (rows - PatchType::Radius) * Rand01(),
                       PatchType::Radius + (cols - PatchType::Radius) * Rand01());


    patches[i].set(I, p);
  }


  for(int i = 0; i < N; ++i)
  {
    auto score = patches[i].zncc(patches[i]);
    THROW_ERROR_IF( score == -1.0 || std::fabs(score - 1.0f) > 1e-6, "bandess" );
  }


  auto t = bpvo::TimeCode(
      5, [&]() {
        for(int i = 0; i < N; ++i) {
        bpvo::ImagePoint p(PatchType::Radius + (rows - PatchType::Radius) * Rand01(),
                       PatchType::Radius + (cols - PatchType::Radius) * Rand01());

        patches[i].set(I, p);
        }
      });

  printf("time %g ms for %d patches [%g ms per point]\n", t, N, t/N);

  return 0;
}

