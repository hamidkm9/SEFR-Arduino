// SEFR: a linear classifier that trains on the microcontroller in a single pass.
//
// Keshavarz, Abadeh, Rawassizadeh. SEFR: A Fast Linear-Time Classifier
// for Ultra-Low Power Devices. https://arxiv.org/abs/2006.04620
//
// Training only keeps per-class feature sums and the range of each feature,
// so samples can be read one at a time from a sensor and thrown away.
// Multiclass problems use one-vs-rest. An optional second pass fits linear
// discriminant analysis on the per-class scores, which usually improves
// multiclass accuracy a lot, and folds the result back into the weights,
// so prediction is still one dot product per class.
//
// All memory is allocated statically from the template sizes:
// about 4 * (2 * N_CLASSES * N_FEATURES + 2 * N_FEATURES + 2 * N_CLASSES * N_CLASSES + 4 * N_CLASSES) bytes.

#ifndef SEFR_H
#define SEFR_H

#include <stdint.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define SEFR_VERSION "1.0.1"

template <uint16_t N_FEATURES, uint16_t N_CLASSES>
class SEFR {
public:
    static_assert(N_FEATURES >= 1, "SEFR needs at least one feature");
    static_assert(N_CLASSES >= 2, "SEFR needs at least two classes");

    SEFR() { reset(); }

    // Forget everything learned so far.
    void reset() {
        memset(sums_, 0, sizeof(sums_));
        memset(counts_, 0, sizeof(counts_));
        memset(weights_, 0, sizeof(weights_));
        memset(biases_, 0, sizeof(biases_));
        for (uint16_t f = 0; f < N_FEATURES; f++) {
            min_[f] = FLT_MAX;
            max_[f] = -FLT_MAX;
        }
        ldaCount_ = 0;
        trained_ = false;
        refining_ = false;
    }

    // Add one training sample. label is the class index, from 0 to N_CLASSES - 1.
    // Returns false if the label is out of range.
    bool learn(const float *features, uint16_t label) {
        if (label >= N_CLASSES) {
            return false;
        }
        float *sums = sums_[label];
        for (uint16_t f = 0; f < N_FEATURES; f++) {
            const float value = features[f];
            sums[f] += value;
            if (value < min_[f]) {
                min_[f] = value;
            }
            if (value > max_[f]) {
                max_[f] = value;
            }
        }
        counts_[label] += 1;
        return true;
    }

    // Compute the weights from the samples learned so far.
    // Can be called again after learning more samples.
    // Returns false unless at least two classes have samples.
    bool train() {
        uint32_t total = 0;
        uint16_t present = 0;
        for (uint16_t c = 0; c < N_CLASSES; c++) {
            total += counts_[c];
            present += counts_[c] > 0 ? 1 : 0;
        }
        refining_ = false;
        if (present < 2) {
            trained_ = false;
            return false;
        }

        for (uint16_t c = 0; c < N_CLASSES; c++) {
            float *weights = weights_[c];
            const uint32_t nPos = counts_[c];
            const uint32_t nNeg = total - nPos;
            if (nPos == 0) {
                memset(weights, 0, sizeof(float) * N_FEATURES);
                biases_[c] = -FLT_MAX;
                continue;
            }
            float scorePos = 0.0f;
            float scoreNeg = 0.0f;
            float offset = 0.0f;
            for (uint16_t f = 0; f < N_FEATURES; f++) {
                const float range = max_[f] - min_[f];
                if (!(range > 0.0f)) {
                    weights[f] = 0.0f;
                    continue;
                }
                float sumAll = 0.0f;
                for (uint16_t k = 0; k < N_CLASSES; k++) {
                    sumAll += sums_[k][f];
                }
                const float meanPos = (sums_[c][f] / (float)nPos - min_[f]) / range;
                const float meanNeg = ((sumAll - sums_[c][f]) / (float)nNeg - min_[f]) / range;
                const float weight = (meanPos - meanNeg) / (meanPos + meanNeg + 1e-7f);
                scorePos += weight * meanPos;
                scoreNeg += weight * meanNeg;
                weights[f] = weight / range;
                offset += weights[f] * min_[f];
            }
            const float threshold = ((float)nNeg * scorePos + (float)nPos * scoreNeg) / (float)total;
            biases_[c] = -(threshold + offset);
        }
        trained_ = true;
        return true;
    }

    // Optional second pass. Call beginRefine() after train(), pass every training
    // sample to refine() again, then call endRefine().
    // Returns false unless the model is trained and every class has samples.
    bool beginRefine() {
        if (!trained_) {
            return false;
        }
        for (uint16_t c = 0; c < N_CLASSES; c++) {
            if (counts_[c] == 0) {
                return false;
            }
        }
        for (uint16_t c = 0; c < N_CLASSES; c++) {
            const float inv = 1.0f / (float)counts_[c];
            for (uint16_t h = 0; h < N_CLASSES; h++) {
                float score = biases_[h];
                for (uint16_t f = 0; f < N_FEATURES; f++) {
                    score += weights_[h][f] * sums_[c][f] * inv;
                }
                ldaMeans_[c][h] = score;
            }
        }
        memset(scatter_, 0, sizeof(scatter_));
        ldaCount_ = 0;
        refining_ = true;
        return true;
    }

    bool refine(const float *features, uint16_t label) {
        if (!refining_ || label >= N_CLASSES) {
            return false;
        }
        float *residual = temp_;
        scores(features, residual);
        for (uint16_t i = 0; i < N_CLASSES; i++) {
            residual[i] -= ldaMeans_[label][i];
        }
        for (uint16_t i = 0; i < N_CLASSES; i++) {
            for (uint16_t j = 0; j <= i; j++) {
                scatter_[i][j] += residual[i] * residual[j];
            }
        }
        ldaCount_ += 1;
        return true;
    }

    bool endRefine() {
        if (!refining_ || ldaCount_ == 0) {
            return false;
        }
        refining_ = false;
        const uint16_t K = N_CLASSES;
        float trace = 0.0f;
        for (uint16_t i = 0; i < K; i++) {
            for (uint16_t j = 0; j <= i; j++) {
                scatter_[i][j] /= (float)ldaCount_;
            }
            trace += scatter_[i][i];
        }
        const float ridge = 1e-6f * trace / (float)K + 1e-12f;
        for (uint16_t i = 0; i < K; i++) {
            scatter_[i][i] += ridge;
        }

        for (uint16_t j = 0; j < K; j++) {
            float diag = scatter_[j][j];
            for (uint16_t k = 0; k < j; k++) {
                diag -= scatter_[j][k] * scatter_[j][k];
            }
            if (!(diag > 0.0f)) {
                return false;
            }
            diag = sqrtf(diag);
            scatter_[j][j] = diag;
            for (uint16_t i = j + 1; i < K; i++) {
                float value = scatter_[i][j];
                for (uint16_t k = 0; k < j; k++) {
                    value -= scatter_[i][k] * scatter_[j][k];
                }
                scatter_[i][j] = value / diag;
            }
        }

        uint32_t total = 0;
        for (uint16_t c = 0; c < K; c++) {
            total += counts_[c];
        }
        float *coef = temp_;
        float *newBias = &temp_[K];
        for (uint16_t c = 0; c < K; c++) {
            float *mean = ldaMeans_[c];
            for (uint16_t i = 0; i < K; i++) {
                float value = mean[i];
                for (uint16_t k = 0; k < i; k++) {
                    value -= scatter_[i][k] * coef[k];
                }
                coef[i] = value / scatter_[i][i];
            }
            for (int32_t i = K - 1; i >= 0; i--) {
                float value = coef[i];
                for (uint16_t k = i + 1; k < K; k++) {
                    value -= scatter_[k][i] * coef[k];
                }
                coef[i] = value / scatter_[i][i];
            }
            float bias = logf((float)counts_[c] / (float)total);
            for (uint16_t i = 0; i < K; i++) {
                bias += coef[i] * (biases_[i] - 0.5f * mean[i]);
            }
            newBias[c] = bias;
            memcpy(mean, coef, sizeof(float) * K);
        }

        for (uint16_t f = 0; f < N_FEATURES; f++) {
            for (uint16_t c = 0; c < K; c++) {
                float value = 0.0f;
                for (uint16_t i = 0; i < K; i++) {
                    value += ldaMeans_[c][i] * weights_[i][f];
                }
                coef[c] = value;
            }
            for (uint16_t c = 0; c < K; c++) {
                weights_[c][f] = coef[c];
            }
        }
        memcpy(biases_, newBias, sizeof(float) * K);
        return true;
    }

    // Return the predicted class index.
    uint16_t predict(const float *features) const {
        float s[N_CLASSES];
        return scores(features, s);
    }

    // Write one score per class into out and return the class with the highest score.
    uint16_t scores(const float *features, float *out) const {
        uint16_t best = 0;
        for (uint16_t c = 0; c < N_CLASSES; c++) {
            float score = biases_[c];
            for (uint16_t f = 0; f < N_FEATURES; f++) {
                score += weights_[c][f] * features[f];
            }
            out[c] = score;
            if (score > out[best]) {
                best = c;
            }
        }
        return best;
    }

    // Use weights trained elsewhere, for example on a PC.
    // weights has N_CLASSES rows of N_FEATURES values.
    void setModel(const float *weights, const float *biases) {
        memcpy(weights_, weights, sizeof(weights_));
        memcpy(biases_, biases, sizeof(biases_));
        trained_ = true;
        refining_ = false;
    }

    const float *weights(uint16_t label) const { return weights_[label]; }
    float bias(uint16_t label) const { return biases_[label]; }
    uint32_t count(uint16_t label) const { return counts_[label]; }
    bool isTrained() const { return trained_; }
    uint16_t numFeatures() const { return N_FEATURES; }
    uint16_t numClasses() const { return N_CLASSES; }

private:
    float sums_[N_CLASSES][N_FEATURES];
    float min_[N_FEATURES];
    float max_[N_FEATURES];
    uint32_t counts_[N_CLASSES];
    float weights_[N_CLASSES][N_FEATURES];
    float biases_[N_CLASSES];
    float ldaMeans_[N_CLASSES][N_CLASSES];
    float scatter_[N_CLASSES][N_CLASSES];
    float temp_[2 * N_CLASSES];
    uint32_t ldaCount_;
    bool trained_;
    bool refining_;
};

#endif
