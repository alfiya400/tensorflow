// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "RunModelViewController.h"

#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>
#import <Foundation/Foundation.h>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/memmapped_file_system.h"

#include "ios_image_load.h"
#include "tensorflow_utils.h"
#define NSLog(FORMAT, ...) printf("%s\n", [[NSString stringWithFormat:FORMAT, ##__VA_ARGS__] UTF8String]);


// If you have your own model, modify this to the file name, and make sure
// you've added the file to your app resources too.
static NSString* model_file_name = @"mmapped";
static NSString* model_file_type = @"pb";
// This controls whether we'll be loading a plain GraphDef proto, or a
// file created by the convert_graphdef_memmapped_format utility that wraps a
// GraphDef and parameter file that can be mapped into memory from file to
// reduce overall memory usage.
const bool model_uses_memory_mapping = true;
// If you have your own model, point this to the labels file.
static NSString* labels_file_name = @"labels";
static NSString* labels_file_type = @"txt";
// These dimensions need to match those the model was trained with.
const int wanted_input_width = 299;
const int wanted_input_height = 299;
const int wanted_input_channels = 3;
const float input_mean = 128.0f;
const float input_std = 128.0f;
const std::string input_layer_name = "input_2";
const std::string output_layer_name = "Softmax_2";


NSString* RunInferenceOnImage();

namespace {
class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
 public:
  explicit IfstreamInputStream(const std::string& file_name)
      : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
  ~IfstreamInputStream() { ifs_.close(); }

  int Read(void* buffer, int size) {
    if (!ifs_) {
      return -1;
    }
    ifs_.read(static_cast<char*>(buffer), size);
    return ifs_.gcount();
  }

 private:
  std::ifstream ifs_;
};
}  // namespace

@interface RunModelViewController ()
@end

@implementation RunModelViewController {
}

- (IBAction)getUrl:(id)sender {
  NSString* inference_result = RunInferenceOnImage();
  self.urlContentTextView.text = inference_result;
}

@end

NSString* RunInferenceOnImage() {
  std::unique_ptr<tensorflow::Session> tf_session;
  std::unique_ptr<tensorflow::MemmappedEnv> tf_memmapped_env;
  std::vector<std::string> labels;
  tensorflow::Status load_status;
  if (model_uses_memory_mapping) {
      load_status = LoadMemoryMappedModel(model_file_name, model_file_type, &tf_session, &tf_memmapped_env);
  } else {
      load_status = LoadModel(model_file_name, model_file_type, &tf_session);
  }
  if (!load_status.ok()) {
      LOG(FATAL) << "Couldn't load model: " << load_status;
  }
  
  tensorflow::Status labels_status =
  LoadLabels(labels_file_name, labels_file_type, &labels);
  if (!labels_status.ok()) {
      LOG(FATAL) << "Couldn't load labels: " << labels_status;
  }
    
  // Read the Grace Hopper image.
  LOG(INFO) << "Loading image";
  NSString* image_path = FilePathForResourceName(@"log", @"jpg");
  int image_width;
  int image_height;
  int image_channels;
  std::vector<tensorflow::uint8> image_data = LoadImageFromFile(
	[image_path UTF8String], &image_width, &image_height, &image_channels);
  assert(image_channels >= wanted_input_channels);
  LOG(INFO) << "Do stuff";
  tensorflow::Tensor image_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({
          1, wanted_input_height, wanted_input_width, wanted_input_channels}));
  auto image_tensor_mapped = image_tensor.tensor<float, 4>();
  tensorflow::uint8* in = image_data.data();
 // tensorflow::uint8* in_end = (in + (image_height * image_width * image_channels));
  float* out = image_tensor_mapped.data();
  NSString *log_str = @"";
  for (int y = 0; y < wanted_input_height; ++y) {
    const int in_y = (y * image_height) / wanted_input_height;
    tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
    float* out_row = out + (y * wanted_input_width * wanted_input_channels);
    for (int x = 0; x < wanted_input_width; ++x) {
      const int in_x = (x * image_width) / wanted_input_width;
      tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
      float* out_pixel = out_row + (x * wanted_input_channels);
      for (int c = 0; c < wanted_input_channels; ++c) {
        out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
        log_str = [log_str stringByAppendingString:[NSString stringWithFormat:@"%u,", in_pixel[c]]];

      }
  //    NSLog(log_str);
      log_str = @"";

    }
  }

  NSString* result = [model_file_name stringByAppendingString: @" - loaded!"];
/*
  result = [NSString stringWithFormat: @"%@ - %d, %s - %dx%d", result,
	label_strings.size(), label_strings[0].c_str(), image_width, image_height];

  std::string input_layer = "Mul";
  std::string output_layer = "final_result";
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status = session->Run({{input_layer, image_tensor}},
				               {output_layer}, {}, &outputs);
*/
  
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status = tf_session->Run({{input_layer_name, image_tensor}},
    {output_layer_name}, {}, &outputs);
      /*
      if (!run_status.ok()) {
          LOG(ERROR) << "Running model failed:" << run_status;
      } else {
          tensorflow::Tensor *output = &outputs[0];
          auto predictions = output->flat<float>();
          
          NSMutableDictionary *newValues = [NSMutableDictionary dictionary];
          for (int index = 0; index < predictions.size(); index += 1) {
              const float predictionValue = predictions(index);
              if (predictionValue > 0.05f) {
                  std::string label = labels[index % predictions.size()];
                  NSString *labelObject = [NSString stringWithCString:label.c_str()];
                  NSNumber *valueObject = [NSNumber numberWithFloat:predictionValue];
                  [newValues setObject:valueObject forKey:labelObject];
              }
          }
          dispatch_async(dispatch_get_main_queue(), ^(void) {
              [self setPredictionValues:newValues];
          });
      }
       */

  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    tensorflow::LogAllRegisteredKernels();
    result = @"Error running model";
    return result;
  }
  tensorflow::string status_string = run_status.ToString();
  result = [NSString stringWithFormat: @"%@ - %s", result,
	status_string.c_str()];

  tensorflow::Tensor* output = &outputs[0];
  const int kNumResults = 5;
  const float kThreshold = 0.1f;
  std::vector<std::pair<float, int> > top_results;
  GetTopN(output->flat<float>(), kNumResults, kThreshold, &top_results);

  std::stringstream ss;
  ss.precision(3);
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;

    ss << index << " " << confidence << "  ";

    // Write out the result as a string
    if (index < labels.size()) {
      // just for safety: theoretically, the output is under 1000 unless there
      // is some numerical issues leading to a wrong prediction.
      ss << labels[index];
    } else {
      ss << "Prediction: " << index;
    }

    ss << "\n";
  }

  LOG(INFO) << "Predictions: " << ss.str();

  tensorflow::string predictions = ss.str();
  result = [NSString stringWithFormat: @"%@ - %s", result,
	predictions.c_str()];

  return result;
}
