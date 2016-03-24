#include "bpvo/imgproc.h"
#include "bpvo/utils.h"
#include "bpvo/timer.h"
#include "utils/kitti_dataset.h"

#include <array>
#include <immintrin.h>

using namespace bpvo;

void print16(const __m128i& v)
{
  uint16_t buf[8];
  _mm_store_si128(  (__m128i*) buf, v );
  printf("%u %u %u %u %u %u %u %u\n",
         buf[0], buf[1], buf[2], buf[3],
         buf[4], buf[5], buf[6], buf[7]);
}

/**
 * \param ptr poitner to the data
 * \param stride the data stride
 * \param r row location of the pixel to test
 * \param c col location of the pixel ot test
 */
inline bool localMax16(const uint16_t* ptr, int stride, int r, int c)
{
  auto p = ptr + r*stride + c;
  auto v0 = *p, v1 = *(p + 4);
  auto v = _mm_setr_epi16(v0, v0, v0, v0,
                          v1, v1, v1, v1);

  auto l0 = _mm_load_si128( (const __m128i*) p - stride - 1),
       l1 = _mm_load_si128( (const __m128i*) p + stride - 1 );

  auto c0 = _mm_cmpgt_epi16( v, l0 ),
       c1 = _mm_cmpgt_epi16( v, l1 );
  printf(" v: "); print16(v);
  printf("l0: "); print16(l0);
  printf("l1: "); print16(l1);
  printf("c0: "); print16(c0);
  printf("c1: "); print16(c1);

  int ok = _mm_test_all_ones( _mm_cmpgt_epi16(v, l0) ) &&
           _mm_test_all_ones( _mm_cmpgt_epi16(v, l1) );

  return ok;
}

int main()
{
  std::array<uint16_t, 3*8> data{
    {1, 2, 3, 4, 5, 6, 7, 8,
     0, 1, 0, 0, 0, 0, 0, 0,
     10, 20, 30, 40, 50, 60, 70, 80}
  };

  localMax16( data.data(), 8, 1, 1);

  return 0;


  KittiDataset dataset("../conf/kitti_seq_0.cfg");
  auto frame = dataset.getFrame(0);

  cv::Mat I;
  frame->image().convertTo(I, CV_32FC1);

  {
    const cv::Mat_<float>& src = (const cv::Mat_<float>&) I;
    cv::Mat_<float> G(src.size());
    double t = TimeCode(1000, [&]() { gradientAbsoluteMagnitude(src, G); });
    printf("time float : %f\n", t);

    IsLocalMax<float> is_local_max(G.ptr<const float>(), G.cols, 1);
    std::vector<cv::Point> uv;
    t = TimeCode(100, [&](){
                 uv.resize(0);
                 uv.reserve(G.rows * G.cols * 0.5);
                 for(int r = 1; r < G.rows - 1; ++r)
                 for(int c = 1; c < G.cols - 1; ++c)
                 if(is_local_max(r,c)) uv.push_back(cv::Point2f(c,r));
                 });
    printf("got %zu\n", uv.size());
    printf("IsLocalMax float %f\n", t);
  }

  {
    cv::Mat_<uint16_t> G(I.size());
    auto sptr = I.ptr<const float>();
    auto dptr = G.ptr<uint16_t>();
    double t = TimeCode(1000, [&]() {gradientAbsoluteMagnitude(sptr, I.rows, I.cols, dptr); } );
    printf("time short : %f\n", t);

    IsLocalMax_u16 is_local_max(G.ptr<const uint16_t>(), G.cols, 1);
    std::vector<cv::Point> uv;
    t = TimeCode(100, [&](){
                 uv.resize(0);
                 uv.reserve(G.rows * G.cols * 0.5);
                 for(int r = 1; r < G.rows - 1; ++r)
                 for(int c = 1; c < G.cols - 1; ++c)
                 if(is_local_max(r,c)) uv.push_back(cv::Point2f(c,r));
                 });
    printf("IsLocalMax short %f\n", t);
  }

  return 0;
}

