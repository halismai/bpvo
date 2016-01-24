#ifndef BPVO_CENSUS_H
#define BPVO_CENSUS_H

namespace cv {
class Mat;
};

namespace bpvo {

/**
 * compute the Census Transform (or 3x3 LBP) on the input image
 *
 * If sigma >=0 a Gaussian filter will be applied to the image before computing
 * the census
 */
cv::Mat census(const cv::Mat& src, float sigma = -1.0f);

}; // bpvo

#endif // BPVO_CENSUS_H
