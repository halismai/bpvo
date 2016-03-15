#include "utils/bounded_buffer.h"
#include "utils/dataset_loader_thread.h"

using namespace bpvo;

int main()
{
  typename DatasetLoaderThread::BufferType buffer(16);
  DatasetLoaderThread data_loader_thread(Dataset::Create("../conf/tsukuba.cfg"), buffer);

  return 0;
}

