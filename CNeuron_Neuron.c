#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PI 3.14159265358979
#define OFF 0
#define ON 1
#define EOL '\0'
#define MAX_PRECISION 1.0E-15
#define MAX_MOMENTUM 1.0
#define UTF8_TKN_BASE 255.0
#define UTF8_NUL_TKN (double) EOF
#define UTF8_PAD_TKN 0.125

typedef struct CNeuron_Result {
    int   code;
    char *error;
} CNeuron_Result;

CNeuron_Result cneuron_result(int code, char *func, char *error) {
    CNeuron_Result x;
    x.code = code;
    x.error = (char *)malloc(128 * sizeof(char));
    x.error[0] = '\0';
    if (x.error != NULL) {
        strcat(x.error, "NeuralC->");
        strcat(x.error, func);
        strcat(x.error, "()->");
        strcat(x.error, error);
    }
    return x;
}

typedef struct CNeuron_Neuron {
    char   *idx;
    int     shape;
    double *weights;
    double *bias;
    double  lrFactor;
    double  nSlope;
    double  momentum;
    double *velocity;
    double  initVel;
    double *losses;
} CNeuron_Neuron;

CNeuron_Result cneuron_neuron_init(CNeuron_Neuron *n, char *idx, int shape, double lrFactor, double nSlope, double velocity, double momentum) {
    char *func = (char *)__FUNCTION__;
    n->idx = idx;
    n->shape = shape < 1 ? 1 : shape > 2 ? shape = shape + (shape % 2) : shape;
    int dShape = n->shape * (size_t)8U;
    n->weights = (double *)malloc(dShape);
    if (n->weights == NULL) {
        return cneuron_result(EOF, func, "");
    }
    n->bias = (double *)malloc(dShape);
    if (n->bias == NULL) {
        return cneuron_result(EOF, func, "");
    }
    n->velocity = (double *)malloc(dShape);
    if (n->velocity == NULL) {
        return cneuron_result(EOF, func, "");
    }
    n->initVel = velocity;
    for (int i = 0; i < n->shape; ++i) {
        n->weights[i] = ((double)rand() / (double)RAND_MAX) * PI;
        n->bias[i] = ((double)rand() / (double)RAND_MAX) * PI;
        n->velocity[i] = velocity;
    }
    n->lrFactor = lrFactor < MAX_PRECISION ? MAX_PRECISION : lrFactor;
    n->nSlope = nSlope < MAX_PRECISION ? MAX_PRECISION : nSlope;
    n->momentum = momentum > MAX_MOMENTUM ? MAX_MOMENTUM : momentum < MAX_PRECISION ? MAX_PRECISION : momentum;
    return cneuron_result(0, func, "");
}

CNeuron_Result cneuron_neuron_forward(CNeuron_Neuron *n, double *in, double *out) {
    char *func = (char *)__FUNCTION__;
    double d = 0.0;
    if (in == NULL) {
        return cneuron_result(EOF, func, "");
    }
    if (out == NULL) {
        return cneuron_result(EOF, func, "");
    }
    for (int i = 0; i < n->shape; ++i) {
        d += in[i] * n->weights[i];
    }
    for (int i = 0; i < n->shape; ++i) {
        double x = d + n->bias[i];
        out[i] = x > 0.0 ? x : n->nSlope * x;
    }
    return cneuron_result(0, func, "");
}

CNeuron_Result cneuron_neuron_train(CNeuron_Neuron *n, double *inputs, double *targets, int lines, int epochs) {
    char *func = (char *)__FUNCTION__;
    if (epochs < 1) {
        epochs = 1;
    }
    double bLoss = 100.0;
    int dShape = n->shape * (size_t)8U;
    double *bWeights = (double *)malloc(dShape);
    if (bWeights == NULL) {
        return cneuron_result(EOF, func, "");
    }
    double *bBias = (double *)malloc(dShape);
    if (bBias == NULL) {
        return cneuron_result(EOF, func, "");
    }
    for (int i = 0; i < n->shape; ++i) {
        bWeights[i] = 0.0;
        bBias[i] = 0.0;
    }
    free(n->losses);
    n->losses = (double *)malloc((size_t)8U);
    if (n->losses == NULL) {
        return cneuron_result(EOF, func, "");
    }
    int dPairs = dShape * lines;
    double *e = (double *)malloc(dPairs);
    if (e == NULL) {
        return cneuron_result(EOF, func, "");
    }
    double *o = (double *)malloc(dPairs);
    if (o == NULL) {
        return cneuron_result(EOF, func, "");
    }
    double *a = (double *)malloc(dPairs);
    if (a == NULL) {
        return cneuron_result(EOF, func, "");
    }
    for (int epoch = 0; epoch < epochs; ++epoch) {
        if (epoch % 1000 == 0) {
            printf("\ntraining -> finished epoch %d/%d", epoch, epochs);
        }
        double loss = 0.0;
        for (int p = 0; p < lines; ++p) {
            int bIdx = p * n->shape;
            for (int i = 0; i < n->shape; ++i) {
                cneuron_neuron_forward(n, &inputs[bIdx], &o[bIdx]);
                int cIdx = bIdx + i;
                e[cIdx] = targets[cIdx] - o[cIdx];
                for (int u = 0; u < n->shape; ++u) {
                    a[bIdx + u] = o[bIdx + u] > 0.0 ? 1.0 : n->nSlope;
                }
                double eIdx = e[cIdx];
                double aIdx = a[cIdx];
                n->velocity[i] = (n->momentum * n->velocity[i]) - (n->lrFactor * -(inputs[bIdx + i] * (eIdx * aIdx)));
                n->weights[i] += n->velocity[i];
                double b = n->bias[i] + (n->velocity[n->shape - 1] * (-eIdx * aIdx));
                n->bias[i] = b > MAX_PRECISION ? MAX_PRECISION : b;
            }
        }
        for (int i = 0; i < n->shape * lines; ++i) {
            double tfIn = targets[i] - o[i];
            loss += 0.5 * (tfIn * tfIn);
        }
        loss /= (n->shape * lines);
        if (loss < bLoss) {
            bLoss = loss;
            for (int i = 0; i < n->shape; ++i) {
                bWeights[i] = n->weights[i];
                bBias[i] = n->bias[i];
            }
        }
    }
    for (int i = 0; i < n->shape; ++i) {
        n->weights[i] = bWeights[i];
        n->bias[i] = bBias[i];
    }
    free(bWeights);
    free(bBias);
    free(e);
    free(o);
    free(a);
    return cneuron_result(0, func, "");
}

CNeuron_Result cneuron_neuron_save(CNeuron_Neuron *n, const char *f) {
    char *func = (char *)__FUNCTION__;
    FILE *fp = fopen(f, "w");
    if (fp == NULL) {
        return cneuron_result(EOF, func, "");
    }
    fprintf(fp, "NeuralC_Neuron %s %d", n->idx, n->shape);
    for (int i = 0; i < n->shape; ++i) {
        fprintf(fp, " %.15f", n->weights[i]);
    }
    for (int i = 0; i < n->shape; ++i) {
        fprintf(fp, " %.15f", n->bias[i]);
    }
    fprintf(fp, " %.15f %.15f %.15f %.15f", n->lrFactor, n->nSlope, n->momentum, n->initVel);
    fclose(fp);
    return cneuron_result(0, func, "");
}

CNeuron_Result cneuron_neuron_load(CNeuron_Neuron *n, char *f, char *neuronName) {
    char *func = (char *)__FUNCTION__;
    FILE *fp = fopen(f, "r");
    if (fp == NULL) {
        return cneuron_result(EOF, func, "");
    }
    if (n->weights != NULL) {
        free(n->weights);
    }
    if (n->bias != NULL) {
        free(n->bias);
    }
    if (n->velocity != NULL) {
        free(n->velocity);
    }
    int dShape = n->shape * (size_t)8U;
    n->weights = (double *)malloc(dShape);
    if (n->weights == NULL) {
        return cneuron_result(EOF, func, "");
    }
    n->bias = (double *)malloc(dShape);
    if (n->bias == NULL) {
        return cneuron_result(EOF, func, "");
    }
    n->velocity = (double *)malloc(dShape);
    if (n->velocity == NULL) {
        return cneuron_result(EOF, func, "");
    }
    n->losses = (double *)malloc((size_t)8U);
    if (n->losses == NULL) {
        return cneuron_result(EOF, func, "");
    }
    char line[(n->shape * 2) + 2];
    while (fgets(line, sizeof(line), fp)) {
        if (strcmp(line, "NeuralC_Neuron") == 0) {
            char idx[32];
            sscanf(line, "NeuralC_Neuron %s", idx);
            if (strcmp(neuronName, idx) == 0) {
                n->idx = idx;
                sscanf(line, " %d", &(n->shape));
                for (int i = 0; i < n->shape; ++i) {
                    fscanf(fp, " %.15f", &(n->weights[i]));
                }
                for (int i = 0; i < n->shape; ++i) {
                    fscanf(fp, " %.15f", &(n->bias[i]));
                }
                sscanf(line, " %.15f %.15f %.15f %.15f", &(n->lrFactor), &(n->nSlope), &(n->momentum), &(n->initVel));
                for (int i = 0; i < n->shape; ++i) {
                    n->velocity[i] = n->initVel;
                }
                break;
            }
        }
    }
    fclose(fp);
    return cneuron_result(0, func, "");
}

int cneuron_count_lines_in_file(const char *f) {
    FILE *fp = fopen(f, "r");
    if (fp == NULL) {
        return EOF;
    }
    int lines = 1;
    int ch;
    int valid = 0;
    while ((ch = fgetc(fp)) != EOF) {
        if (valid == 1) {
            if (ch == (int)'|') {
                ++valid;
                continue;
            }
        }
        if (ch == (int)'\n') {
            ++lines;
        }
    }
    fclose(fp);
    return lines;
}

CNeuron_Result cneuron_get_utf8_dataset(const char *f, const int shape, double *input, double *target) {
    char *func = (char *)__FUNCTION__;
    FILE *fp = fopen(f, "r");
    if (fp == NULL) {
        return cneuron_result(EOF, func, "");
    }
    int size = 1;
    int allocSize = (shape * 2) + 2;
    int allocated = allocSize;
    int used = 0;
    double *tokens = (double *)malloc(allocated * (size_t)8U);
    if (tokens == NULL) {
        fclose(fp);
        return cneuron_result(EOF, func, "");
    }
    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        if (used >= allocated) {
            allocated += allocSize;
            size += 1;
            double *reallocAttempt = (double *)realloc(tokens, allocated * (size_t)8U);
            if (reallocAttempt == NULL) {
                fclose(fp);
                free(tokens);
                return cneuron_result(EOF, func, "");
            }
            tokens = reallocAttempt;
        }
        if (ch == (int)'\n' || ch == (int)'|') {
            while (used % shape != 0) {
                tokens[used++] = UTF8_PAD_TKN;
            }
        } else {
            tokens[used++] = (double)ch / UTF8_TKN_BASE;
        }
    }
    while (used % shape != 0) {
        tokens[used++] = UTF8_PAD_TKN;
    }
    tokens[used] = UTF8_NUL_TKN;
    fclose(fp);
    for (int i = 0; i < size; ++i) {
        for (int u = 0; u < shape; ++u) {
            input[i * shape + u] = tokens[i * (2 * shape) + u];
            target[i * shape + u] = tokens[i * (2 * shape) + shape + u];
        }
    }
    return cneuron_result(0, func, "");
}
