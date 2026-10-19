#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../../src/SEFR.h"

template <uint16_t F, uint16_t K>
int run(const char *path, const char *out, bool lda) {
    FILE *in = fopen(path, "r");
    std::vector<float> X;
    std::vector<int> y;
    int label;
    while (fscanf(in, "%d", &label) == 1) {
        y.push_back(label);
        for (int f = 0; f < F; f++) {
            float v;
            fscanf(in, ",%f", &v);
            X.push_back(v);
        }
    }
    fclose(in);
    static SEFR<F, K> model;
    model.reset();
    for (size_t i = 0; i < y.size(); i++) model.learn(&X[i * F], y[i]);
    if (!model.train()) return 1;
    if (lda) {
        if (!model.beginRefine()) return 2;
        for (size_t i = 0; i < y.size(); i++) model.refine(&X[i * F], y[i]);
        if (!model.endRefine()) return 3;
    }
    FILE *o = fopen(out, "w");
    for (int c = 0; c < K; c++) {
        for (int f = 0; f < F; f++) fprintf(o, "%.9g,", model.weights(c)[f]);
        fprintf(o, "%.9g\n", model.bias(c));
    }
    for (size_t i = 0; i < y.size(); i++) fprintf(o, "%d\n", model.predict(&X[i * F]));
    fclose(o);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 5) return 8;
    bool lda = atoi(argv[3]);
    const char *name = argv[4];
    if (!strcmp(name, "iris")) return run<4, 3>(argv[1], argv[2], lda);
    if (!strcmp(name, "wine")) return run<13, 3>(argv[1], argv[2], lda);
    if (!strcmp(name, "cancer")) return run<30, 2>(argv[1], argv[2], lda);
    if (!strcmp(name, "digits")) return run<64, 10>(argv[1], argv[2], lda);
    return 9;
}
