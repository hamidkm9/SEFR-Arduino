// Teach the board new examples over the serial port, then ask it to classify.
//
// Open the Serial Monitor at 115200 baud, set it to send a newline, and type:
//   learn 0 120 80 40    adds a sample of class 0 with three feature values
//   train                computes the model from the samples so far
//   predict 118 82 39    prints the predicted class and the score of each class
//   reset                forgets everything
//
// Replace the typed values with readings from a sensor, for example the red,
// green and blue channels of a color sensor, to train it on real objects.

#include <SEFR.h>

const uint16_t FEATURES = 3;
const uint16_t CLASSES = 4;

SEFR<FEATURES, CLASSES> model;
char line[64];
uint8_t length = 0;

bool parseValues(char *text, float *values, uint16_t count) {
  for (uint16_t i = 0; i < count; i++) {
    char *token = strtok(i == 0 ? text : NULL, " ,");
    if (token == NULL) {
      return false;
    }
    values[i] = atof(token);
  }
  return true;
}

void handle(char *command) {
  float features[FEATURES];
  if (strncmp(command, "learn ", 6) == 0) {
    char *rest = command + 6;
    char *labelText = strtok(rest, " ,");
    if (labelText == NULL || !parseValues(NULL, features, FEATURES)) {
      Serial.println("usage: learn <class> <f1> <f2> <f3>");
      return;
    }
    if (!model.learn(features, atoi(labelText))) {
      Serial.println("class must be between 0 and 3");
      return;
    }
    Serial.println("ok");
  } else if (strcmp(command, "train") == 0) {
    Serial.println(model.train() ? "trained" : "need samples of at least two classes");
  } else if (strncmp(command, "predict ", 8) == 0) {
    if (!model.isTrained()) {
      Serial.println("train first");
      return;
    }
    if (!parseValues(command + 8, features, FEATURES)) {
      Serial.println("usage: predict <f1> <f2> <f3>");
      return;
    }
    float scores[CLASSES];
    uint16_t label = model.scores(features, scores);
    Serial.print("class ");
    Serial.print(label);
    Serial.print("  scores:");
    for (uint16_t c = 0; c < CLASSES; c++) {
      Serial.print(' ');
      Serial.print(scores[c], 3);
    }
    Serial.println();
  } else if (strcmp(command, "reset") == 0) {
    model.reset();
    Serial.println("reset");
  } else if (length > 0) {
    Serial.println("commands: learn, train, predict, reset");
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
  }
  Serial.println("SEFR serial trainer ready");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      line[length] = '\0';
      handle(line);
      length = 0;
    } else if (length < sizeof(line) - 1) {
      line[length++] = c;
    }
  }
}
