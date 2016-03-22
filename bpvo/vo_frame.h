#ifndef BPVO_VO_FRAME_H
#define BPVO_VO_FRAME_H

#include <bpvo/types.h>

#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

namespace cv {
class Mat;
}; // cv

namespace bpvo {

class TemplateData;
class DenseDescriptor;
class DenseDescriptorPyramid;

class VisualOdometryFrame
{
  typedef UniquePointer<TemplateData> TemplateDataPointer;

 public:
  /**
   */
  VisualOdometryFrame(const Matrix33& K, float b, const AlgorithmParameters&);

  /**
   */
  ~VisualOdometryFrame();

  VisualOdometryFrame(const VisualOdometryFrame&) = delete;
  VisualOdometryFrame& operator=(const VisualOdometryFrame&) = delete;

  /**
   */
  void setData(const cv::Mat& image, const cv::Mat& disparity);

  /**
   */
  void setDataAndTemplate(const cv::Mat&, const cv::Mat&);

  /**
   * compute the template data
   */
  void setTemplate();

  inline void clear() { _has_data = false; _has_template = false; }
  inline bool empty() { return !_has_data; }

  inline bool hasTemplate() const { return _has_template; }

  /**
   * \return the dense descriptor at specified pyramid level
   */
  const DenseDescriptor* getDenseDescriptorAtLevel(size_t) const;

  /**
   * \return the template data at the specified pyramid level
   */
  const TemplateData* getTemplateDataAtLevel(size_t) const;

  /**
   * \return the number of pyramid levels
   */
  int numLevels() const;

  /**
   * \return pointer to the raw input image
   */
  const cv::Mat* imagePointer() const;

  /**
   * \return pointer to the raw disparity
   */
  const cv::Mat* disparityPointer() const;


 private:
  int _max_test_level;
  bool _has_data;
  bool _has_template;
  UniquePointer<cv::Mat> _image;
  UniquePointer<cv::Mat> _disparity;
  UniquePointer<DenseDescriptorPyramid> _desc_pyr;
  std::vector<TemplateDataPointer> _tdata_pyr;
}; // VisualOdometryFrame

}; // bpvo

#endif // BPVO_VO_FRAME_H
