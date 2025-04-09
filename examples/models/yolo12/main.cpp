#include "inference.h"

#include <gflags/gflags.h>

void draw_detection(
    cv::Mat& frame,
    const Detection detection,
    const cv::Scalar color);

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

DEFINE_string(input_path, "input.mp4", "Path to the input video");

DEFINE_string(output_path, "output.mp4", "Path to the output video");

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Module yolo_module(FLAGS_model_path);

  auto error = yolo_module.load();
  error = yolo_module.load_forward();

  const auto model_input_shape =
      yolo_module.method_meta("forward")->input_tensor_meta(0)->sizes();
  const cv::Size img_dims = {model_input_shape[3], model_input_shape[2]};

  cv::VideoCapture cap(FLAGS_input_path.c_str());
  if (!cap.isOpened()) {
    std::cout << "Error opening video stream or file" << std::endl;
    return -1;
  }
  const auto frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  const auto frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  cv::VideoWriter video(
      FLAGS_output_path.c_str(),
      cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
      30,
      cv::Size(frame_width, frame_height));

  while (true) {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty())
      break;

    std::vector<Detection> output =
        infer_yolo_once(yolo_module, frame, img_dims, DEFAULT_YOLO_CONFIG);

    std::cout << "Number of detections:" << output.size() << std::endl;

    for (auto& detection : output) {
      draw_detection(frame, detection, cv::Scalar(0, 0, 255));
    }

    video.write(frame);
  }
  cap.release();
  video.release();
}

void draw_detection(
    cv::Mat& frame,
    const Detection detection,
    const cv::Scalar color) {
  cv::Rect box = detection.box;

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