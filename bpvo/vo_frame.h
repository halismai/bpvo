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

  inline void clear() { _has_data = false; }
  inline bool empty() { return !_has_data; }

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


  inline bool isTemplateDataReady() const { return _is_tdata_ready; }

  inline std::mutex& mutex() { return _mutex; }

 private:
  int _max_test_level;
  bool _has_data;
  UniquePointer<cv::Mat> _disparity;
  UniquePointer<DenseDescriptorPyramid> _desc_pyr;
  std::vector<TemplateDataPointer> _tdata_pyr;

  mutable std::mutex _mutex;
  std::condition_variable _has_template_data;
  std::thread _set_template_thread;

  void set_template_data();

  bool _is_tdata_ready;
  bool _should_quit;
}; // VisualOdometryFrame

}; // bpvo

#endif // BPVO_VO_FRAME_H
