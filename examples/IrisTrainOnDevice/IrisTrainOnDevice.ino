// Train SEFR on the board itself and test it on samples it has not seen.
//
// Every fifth Iris sample is kept for testing and the rest are used for training.
// The model is trained in one pass, then refined with a second pass (LDA),
// and the test accuracy and timings are printed after each step.
// Works on boards with 2 KB of RAM, such as the Arduino Uno:
// the dataset stays in flash and the model needs about 250 bytes.

#include <SEFR.h>
#include "iris_data.h"

SEFR<4, 3> model;

void readSample(uint16_t index, float *features) {
  for (uint8_t f = 0; f < 4; f++) {
    features[f] = pgm_read_float(&IRIS_X[index * 4 + f]);
  }
}

uint8_t readLabel(uint16_t index) {
  return pgm_read_byte(&IRIS_Y[index]);
}

bool isTestSample(uint16_t index) {
  return index % 5 == 0;
}

float testAccuracy() {
  float features[4];
  uint16_t correct = 0;
  uint16_t total = 0;
  for (uint16_t i = 0; i < IRIS_SAMPLES; i++) {
    if (!isTestSample(i)) {
      continue;
    }
    readSample(i, features);
    correct += model.predict(features) == readLabel(i) ? 1 : 0;
    total += 1;
  }
  return 100.0f * correct / total;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
  }

  float features[4];

  unsigned long start = micros();
  for (uint16_t i = 0; i < IRIS_SAMPLES; i++) {
    if (isTestSample(i)) {
      continue;
    }
    readSample(i, features);
    model.learn(features, readLabel(i));
  }
  model.train();
  unsigned long trainTime = micros() - start;

  Serial.print("SEFR, one pass:      test accuracy ");
  Serial.print(testAccuracy(), 1);
  Serial.print("%, training took ");
  Serial.print(trainTime);
  Serial.println(" us");

  start = micros();
  model.beginRefine();
  for (uint16_t i = 0; i < IRIS_SAMPLES; i++) {
    if (isTestSample(i)) {
      continue;
    }
    readSample(i, features);
    model.refine(features, readLabel(i));
  }
  model.endRefine();
  unsigned long refineTime = micros() - start;

  Serial.print("SEFR + LDA refining: test accuracy ");
  Serial.print(testAccuracy(), 1);
  Serial.print("%, refining took ");
  Serial.print(refineTime);
  Serial.println(" us");

  readSample(0, features);
  start = micros();
  uint16_t label = model.predict(features);
  unsigned long predictTime = micros() - start;
  Serial.print("One prediction took ");
  Serial.print(predictTime);
  Serial.print(" us, predicted class ");
  Serial.println(label);
}

void loop() {
}
