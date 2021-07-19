
/* ALL Arduino Nano 33 BLE Sense Classifier

An experiment to explore how low powered microcontrollers, specifically the
Arduino Nano 33 BLE Sense, can be used to detect Acute Lymphoblastic Leukemia.

MIT License

Copyright (c) 2021 Asociación de Investigacion en Inteligencia Artificial
Para la Leucemia Peter Moss

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contributors:
- Adam Milton-Barker
==============================================================================*/

#include "Arduino.h"
#include <SPI.h>

#include <TensorFlowLite.h>

#include "main_functions.h"
#include "all_model.h"
#include "model_settings.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <JPEGDecoder.h>

String images[]={
    "Im006_1.jpg",
    "Im020_1.jpg",
    "Im024_1.jpg",
    "Im026_1.jpg",
    "Im028_1.jpg",
    "Im031_1.jpg",
    "Im035_0.jpg",
    "Im041_0.jpg",
    "Im047_0.jpg",
    "Im053_1.jpg",
    "Im057_1.jpg",
    "Im060_1.jpg",
    "Im063_1.jpg",
    "Im069_0.jpg",
    "Im074_0.jpg",
    "Im088_0.jpg",
    "Im095_0.jpg",
    "Im099_0.jpg",
    "Im101_0.jpg",
    "Im106_0.jpg"
};

int tp = 0;
int fp = 0;
int tn = 0;
int fn = 0;

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  constexpr int kTensorArenaSize = 136 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
} 

void setup() {
  
  Serial.begin(9600);
  while (!Serial) {
    ; 
  }

  Serial.println(F("Initialising SD card..."));
  if (!SD.begin(10)) {
    Serial.println(F("Initialisation failed!"));
    return;
  }
  Serial.println(F("Initialisation done."));
  
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(all_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  
  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  getInputInfo(input);

  for (int i = 0; i < 20; i++) {
    getImage(images[i], input->data.int8);
    TfLiteTensor* output = interpreter->output(0);
    int8_t all_score = output->data.int8[kAllIndex];
    int8_t no_all_score = output->data.int8[kNotAllIndex];
    processScores(all_score, no_all_score, images[i]);
    delay(2000); 
  }

  Serial.print("True Positives: ");
  Serial.println(tp);
  Serial.print("False Positives: ");
  Serial.println(fp);
  Serial.print("True Negatives: ");
  Serial.println(tn);
  Serial.print("False Negatives: ");
  Serial.println(fn);
}

void getInputInfo(TfLiteTensor* input){
  Serial.println("");
  Serial.println("Model input info");
  Serial.println("===============");
  Serial.print("Dimensions: ");
  Serial.println(input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(input->dims->data[1]);
  Serial.print("Dim 3 size: ");
  Serial.println(input->dims->data[2]);
  Serial.print("Dim 4 size: ");
  Serial.println(input->dims->data[3]);
  Serial.print("Input type: ");
  Serial.println(input->type);
  Serial.println("===============");
  Serial.println("");
}

TfLiteStatus getImage(String filepath, int8_t* image_data){
  File jpegFile = SD.open(filepath, FILE_READ);  
  
  if ( !jpegFile ) {
    Serial.print("ERROR: File not found!");
    return kTfLiteError;
  }

  boolean decoded = JpegDec.decodeSdFile(jpegFile);
  processImage(filepath, image_data);

  return kTfLiteOk;
}

void processImage(String filename, int8_t* image_data){

  // Crop the image by keeping a certain number of MCUs in each dimension
  const int keep_x_mcus = kNumCols / JpegDec.MCUWidth;
  const int keep_y_mcus = kNumRows / JpegDec.MCUHeight;

  // Calculate how many MCUs we will throw away on the x axis
  const int skip_x_mcus = JpegDec.MCUSPerRow - keep_x_mcus;
  // Roughly center the crop by skipping half the throwaway MCUs at the
  // beginning of each row
  const int skip_start_x_mcus = skip_x_mcus / 2;
  // Index where we will start throwing away MCUs after the data
  const int skip_end_x_mcu_index = skip_start_x_mcus + keep_x_mcus;
  // Same approach for the columns
  const int skip_y_mcus = JpegDec.MCUSPerCol - keep_y_mcus;
  const int skip_start_y_mcus = skip_y_mcus / 2;
  const int skip_end_y_mcu_index = skip_start_y_mcus + keep_y_mcus;

  // Pointer to the current pixel
  uint16_t* pImg;
  // Color of the current pixel
  uint16_t color;

  // Loop over the MCUs
  while (JpegDec.read()) {
    // Skip over the initial set of rows
    if (JpegDec.MCUy < skip_start_y_mcus) {
      continue;
    }
    // Skip if we're on a column that we don't want
    if (JpegDec.MCUx < skip_start_x_mcus ||
        JpegDec.MCUx >= skip_end_x_mcu_index) {
      continue;
    }
    // Skip if we've got all the rows we want
    if (JpegDec.MCUy >= skip_end_y_mcu_index) {
      continue;
    }
    // Pointer to the current pixel
    pImg = JpegDec.pImage;

    // The x and y indexes of the current MCU, ignoring the MCUs we skip
    int relative_mcu_x = JpegDec.MCUx - skip_start_x_mcus;
    int relative_mcu_y = JpegDec.MCUy - skip_start_y_mcus;

    // The coordinates of the top left of this MCU when applied to the output
    // image
    int x_origin = relative_mcu_x * JpegDec.MCUWidth;
    int y_origin = relative_mcu_y * JpegDec.MCUHeight;

    // Loop through the MCU's rows and columns
    for (int mcu_row = 0; mcu_row < JpegDec.MCUHeight; mcu_row++) {
      // The y coordinate of this pixel in the output index
      int current_y = y_origin + mcu_row;
      for (int mcu_col = 0; mcu_col < JpegDec.MCUWidth; mcu_col++) {
        // Read the color of the pixel as 16-bit integer
        color = *pImg++;
        // Extract the color values (5 red bits, 6 green, 5 blue)
        uint8_t r, g, b;
        r = ((color & 0xF800) >> 11) * 8;
        g = ((color & 0x07E0) >> 5) * 4;
        b = ((color & 0x001F) >> 0) * 8;
        // Convert to grayscale by calculating luminance
        // See https://en.wikipedia.org/wiki/Grayscale for magic numbers
        float gray_value = (0.2126 * r) + (0.7152 * g) + (0.0722 * b);

        // Convert to signed 8-bit integer by subtracting 128.
        gray_value -= 128;
        // The x coordinate of this pixel in the output image
        int current_x = x_origin + mcu_col;
        // The index of this pixel in our flat output buffer
        int index = (current_y * kNumCols) + current_x;
        image_data[index] = static_cast<int8_t>(gray_value);
      }
    }
  }
}

void processScores(int8_t all_score, int8_t no_all_score, String filename){

  Serial.println(filename);
  Serial.println("===============");
  Serial.print("ALL positive score: ");
  Serial.println(all_score);
  Serial.print("ALL negative score: ");
  Serial.println(no_all_score);
  if(all_score > no_all_score && filename.indexOf("_1") > 0){
    Serial.println("True Positive");
    tp = tp + 1;
  }
  else if(all_score > no_all_score && filename.indexOf("_0") > 0){
    Serial.println("False Positive");
    fp = fp + 1;
  }
  else if(all_score < no_all_score && filename.indexOf("_1") > 0){
    Serial.println("False Negative");
    fn = fn + 1;
  }
  else if(all_score < no_all_score && filename.indexOf("_0") > 0){
    Serial.println("True Negative");
    tn = tn + 1;
  }
  Serial.println("");

  static bool is_initialized = false;
  if (!is_initialized) {
    pinMode(LEDR, OUTPUT);
    pinMode(LEDG, OUTPUT);
    pinMode(LEDB, OUTPUT);
    is_initialized = true;
  }
  
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDR, HIGH);
  
  digitalWrite(LEDB, LOW);
  delay(100);
  digitalWrite(LEDB, HIGH);
  
  if (all_score > no_all_score) {
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDR, LOW);
    delay(200);
    digitalWrite(LEDR, HIGH);
  } else {
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDG, LOW);
    delay(200);
    digitalWrite(LEDG, HIGH);
  }

}

void jpegInfo() {

  Serial.println("JPEG image info");
  Serial.println("===============");
  Serial.print("Width      :");
  Serial.println(JpegDec.width);
  Serial.print("Height     :");
  Serial.println(JpegDec.height);
  Serial.print("Components :");
  Serial.println(JpegDec.comps);
  Serial.print("MCU / row  :");
  Serial.println(JpegDec.MCUSPerRow);
  Serial.print("MCU / col  :");
  Serial.println(JpegDec.MCUSPerCol);
  Serial.print("Scan type  :");
  Serial.println(JpegDec.scanType);
  Serial.print("MCU width  :");
  Serial.println(JpegDec.MCUWidth);
  Serial.print("MCU height :");
  Serial.println(JpegDec.MCUHeight);
  Serial.println("===============");
  Serial.println("");
}

void loop() {
}
