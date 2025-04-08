#include "inference.h"

#include <gflags/gflags.h>
#include <csignal>

void draw_detection(cv::Mat& frame, const Detection detection);

DetectionConfig DEFAULT_YOLO_CONFIG = {
    {"person",        "bicycle",      "car",
     "motorcycle",    "airplane",     "bus",
     "train",         "truck",        "boat",
     "traffic light", "fire hydrant", "stop sign",
     "parking meter", "bench",        "bird",
     "cat",           "dog",          "horse",
     "sheep",         "cow",          "elephant",
     "bear",          "zebra",        "giraffe",
     "backpack",      "umbrella",     "handbag",
     "tie",           "suitcase",     "frisbee",
     "skis",          "snowboard",    "sports ball",
     "kite",          "baseball bat", "baseball glove",
     "skateboard",    "surfboard",    "tennis racket",
     "bottle",        "wine glass",   "cup",
     "fork",          "knife",        "spoon",
     "bowl",          "banana",       "apple",
     "sandwich",      "orange",       "broccoli",
     "carrot",        "hot dog",      "pizza",
     "donut",         "cake",         "chair",
     "couch",         "potted plant", "bed",
     "dining table",  "toilet",       "tv",
     "laptop",        "mouse",        "remote",
     "keyboard",      "cell phone",   "microwave",
     "oven",          "toaster",      "sink",
     "refrigerator",  "book",         "clock",
     "vase",          "scissors",     "teddy bear",
     "hair drier",    "toothbrush"},
    0.45,
    0.50};

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(input_path, "input.jpg", "Path to the input image");

DEFINE_string(output_path, "output.jpg", "Path to the output image");

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Module yolo_module(FLAGS_model_path);

  auto error = yolo_module.load();
  // TODO: enable
  // ET_CHECK_MSG(
  //    error == Error::ok ,
  //    "Failed to parse model file %s, error: 0x%",
  //    projectBasePath.c_str(), error);

  const auto names = yolo_module.method_names();
  const auto p = yolo_module.program();
  //const auto ep = executorch::runtime::get_execution_plan(ip, "forward");
  error = yolo_module.load_forward();
  // ET_CHECK_MSG(
  //     error == Error::ok ,
  //     "Failed to get method_meta for forward, error: 0x%", error);

  std::vector<std::string> imageNames({FLAGS_input_path});

  for (int i = 0; i < imageNames.size(); ++i) {
    cv::Mat frame = cv::imread(imageNames[i]);

    std::vector<Detection> output =
        infer_yolo_once(yolo_module, frame, DEFAULT_YOLO_CONFIG);

    std::cout << "Number of detections:" << output.size() << std::endl;

    for (auto& detection : output) {
      draw_detection(frame, detection);
    }

    // This is only for preview purposes
    float scale = 0.8;
    cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
    // std::raise(SIGINT);
    cv::imwrite(FLAGS_output_path, frame);
    // cv::imshow("Inference", frame);
    // cv::waitKey(-1);
  }
}

void draw_detection(cv::Mat& frame, const Detection detection) {
  cv::Rect box = detection.box;
  cv::Scalar color = detection.color;

  // Detection box
  cv::rectangle(frame, box, color, 2);

  // Detection box text
  std::string classString = detection.className + ' ' +
      std::to_string(detection.confidence).substr(0, 4);
  cv::Size textSize =
      cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
  cv::Rect textBox(
      box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

  cv::rectangle(frame, textBox, color, cv::FILLED);
  cv::putText(
      frame,
      classString,
      cv::Point(box.x + 5, box.y - 10),
      cv::FONT_HERSHEY_DUPLEX,
      1,
      cv::Scalar(0, 0, 0),
      2,
      0);
}