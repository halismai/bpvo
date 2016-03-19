#ifndef BPVO_LUCID_H
#define BPVO_LUCID_H

namespace cv { class Mat; };

namespace bpvo {

class LucidDescriptor
{
 public:
  LucidDescriptor(int radius = 1, int blur_radius = 2);

  void compute(const cv::Mat& src, cv::Mat&) const;

 private:
  int _radius;
  int _blur_radius;
}; // LucidDescriptor

}; // bpvo

#endif // BPVO_LUCID_H


