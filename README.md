# SEFR for Arduino

SEFR is a linear classifier that trains on the microcontroller itself. It reads each training sample once, keeps only a running sum per class and the range of each feature, and can then classify new samples with one dot product per class. There is no learning rate, no iterations and no dynamic memory, so it runs on boards with 2 KB of RAM such as the Arduino Uno.

STMicroelectronics ships SEFR in [NanoEdge AI Studio](https://dev.st.com/edge-ai-docs/general/latest/en/docs/misc/nanoedge-ai-index.html), its AutoML tool for STM32, where it is recommended for multi-class classification with an ultra-low footprint.

## Install

In the Arduino IDE, open **Library Manager**, search for **SEFR** and click Install. With the command line:

```sh
arduino-cli lib install SEFR
```

## Example

```cpp
#include <SEFR.h>

SEFR<3, 4> model;           // 3 features, 4 classes

void setup() {
  float sample[3] = {120, 80, 40};
  model.learn(sample, 2);   // add a sample of class 2; repeat for every sample
  model.train();            // compute the weights

  float reading[3] = {118, 82, 39};
  uint16_t label = model.predict(reading);
}
```

Training can continue: call `learn()` with more samples and `train()` again at any time.

For multiclass problems, an optional second pass usually improves accuracy a lot. After `train()`, call `beginRefine()`, pass every training sample to `refine()` once more, and call `endRefine()`. It fits linear discriminant analysis on the class scores and folds it into the same weights, so prediction stays exactly as fast.

## Examples

| Example | What it shows |
|---|---|
| `IrisTrainOnDevice` | Trains on the board from data stored in flash, then tests on held-out samples, with and without the refining pass. Fits on an Uno. |
| `SerialTrainer` | Teach the board new samples over the Serial Monitor, then ask it to classify. A starting point for training on sensor readings. |
| `PretrainedModel` | Loads weights trained elsewhere with `setModel()`. |

## API

| Method | Description |
|---|---|
| `SEFR<N_FEATURES, N_CLASSES> model;` | Creates a model. All memory is allocated statically. |
| `bool learn(const float *x, uint16_t label)` | Adds one sample. Labels go from 0 to `N_CLASSES - 1`. |
| `bool train()` | Computes weights from the samples so far. Needs samples of at least two classes. |
| `bool beginRefine()`, `bool refine(x, label)`, `bool endRefine()` | Optional second pass over the same samples. Needs samples of every class. |
| `uint16_t predict(const float *x)` | Returns the predicted class. |
| `uint16_t scores(const float *x, float *out)` | Writes one score per class and returns the best class. |
| `void setModel(const float *weights, const float *biases)` | Loads weights (`N_CLASSES` rows of `N_FEATURES`) and biases trained elsewhere. |
| `weights(c)`, `bias(c)`, `count(c)` | Read the weights, bias and number of samples of class `c`. |
| `reset()` | Forgets everything. |

## Memory

A model takes about `4 × (2·N_CLASSES·N_FEATURES + 2·N_FEATURES + 2·N_CLASSES² + 4·N_CLASSES)` bytes. For example, 4 features and 3 classes take about 250 bytes, and 64 features and 10 classes take about 6.6 KB.

## How it works

For each class, SEFR scales every feature to [0, 1], compares the average of the class with the average of all other classes, and sets the weight of each feature to their normalised difference. The threshold sits between the average scores of the two groups, weighted by their sizes. The paper describes the binary version, and this library uses one-vs-rest for more classes. The scaling is folded into the weights, so raw sensor values can be passed to `predict()`.

The library is tested against an independent NumPy implementation on the Iris, Wine, Breast Cancer and Digits datasets; see `extras/test`.

## Reference

Hamidreza Keshavarz, Mohammad Saniee Abadeh, Reza Rawassizadeh. *SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices.* [arXiv:2006.04620](https://arxiv.org/abs/2006.04620), 2020.

## License

MIT
