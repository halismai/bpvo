#include <utils/tsukuba_dataset.h>
using namespace bpvo;

int main()
{
  UniquePointer<Dataset> dataset = UniquePointer<Dataset>(
      new TsukubaSyntheticDataset("../conf/tsukuba.cfg"));

  for(int i = 0; i < 10; ++i)
  {
    auto f = dataset->getFrame(i);
  }

  return 0;
}
