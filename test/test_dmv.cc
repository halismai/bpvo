#include <cstdio>

#if defined(WITH_DMV)

#include <dmv/photo_bundle.h>
#include <dmv/photo_bundle_config.h>

#include <bpvo/timer.h>
#include <bpvo/utils.h>

#include <bpvo/vo_output_reader.h>
#include <bpvo/vo_output.h>

#include <iostream>

using namespace bpvo;
using namespace bpvo::dmv;

int main()
{
  Matrix33 K;
  K << 615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0;

  PhotoBundleConfig config;
  PhotoBundle photo_bundle(K, config);

  VoOutputReader vo_output_reader(".", "vo_%05d.voout", 1);

  double total_time = 0.0;
  int i= 0;
  for(i = 0; i < config.bundleWindowSize + 1; ++i)
  {
    auto frame = vo_output_reader[i];
    if(frame) {
      Timer timer;
      photo_bundle.addData(frame.get());
      total_time += timer.stop().count() / 1000.0;
    } else
      break;
  }

  printf("time %f for %d frames\n", total_time,  i);
  printf("total mem usage %0.2f MiB\n", procMemUsage() * 0.000953674);

  std::cout << "Press any key to continue...\n";
  std::cin.get();
}
#else
int main()
{
  printf("compile WITH_DMV\n");
}
#endif
