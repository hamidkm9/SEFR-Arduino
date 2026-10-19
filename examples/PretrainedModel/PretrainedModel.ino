// Run a SEFR model that was trained somewhere else.
//
// The weights below were learned by this library from all 150 Iris samples,
// with the LDA refining pass. They could also come from a PC, or from another
// board that printed weights(c) and bias(c) after training.

#include <SEFR.h>

const float WEIGHTS[3][4] = {
  {-288.354156f, 80.9543381f, 155.331955f, 317.348572f},
  {-292.952423f, 61.5881729f, 171.191895f, 350.245392f},
  {-295.939789f, 57.8896446f, 178.302048f, 366.005737f},
};
const float BIASES[3] = {6937.04639f, 6950.98682f, 6919.02295f};

const char *NAMES[3] = {"setosa", "versicolor", "virginica"};

SEFR<4, 3> model;

void classify(float sepalLength, float sepalWidth, float petalLength, float petalWidth) {
  float features[4] = {sepalLength, sepalWidth, petalLength, petalWidth};
  uint16_t label = model.predict(features);
  Serial.print("predicted: ");
  Serial.println(NAMES[label]);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
  }

  model.setModel(&WEIGHTS[0][0], BIASES);

  classify(5.0f, 3.4f, 1.5f, 0.2f);
  classify(6.0f, 2.8f, 4.4f, 1.3f);
  classify(6.7f, 3.1f, 5.8f, 2.2f);
}

void loop() {
}
