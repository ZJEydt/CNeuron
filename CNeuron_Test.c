#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "./CNeuron_Neuron.c"

int test_function() {

    // Create a CNeuron_Neuron
    CNeuron_Neuron myNeuron;
    int neuronShape = 16;
    char *idx = "myNeuron";
    CNeuron_Result neuronInit = cneuron_neuron_init(&myNeuron, idx, neuronShape, 1.0E-5, 1.0E-5, 1.0E-5, 0.9995);
    if (neuronInit.code == EOF) {
        printf("\n");
        return EOF;
    }

    // Save & Load the CNeuron_Neuron
    char *neuronFile = "./_CNeuron_Neuron_Test.txt";
    CNeuron_Result neuronSave = cneuron_neuron_save(&myNeuron, neuronFile);
    if (neuronSave.code == EOF) {
        printf("\n%s\n", neuronSave.error);
        return EOF;
    }
    CNeuron_Result neuronLoad = cneuron_neuron_load(&myNeuron, neuronFile, idx);
    if (neuronLoad.code == EOF) {
        printf("\n%s\n", neuronLoad.error);
        return EOF;
    }

    // Load a `UTF-8` Prompt Dataset from a file
    const char *datasetFile = "./_CNeuron_UTF8_Dataset_Test.txt";
    int datasetLines = cneuron_count_lines_in_file(datasetFile);
    if (datasetLines == EOF) {
        printf("\n");
        return EOF;
    }
    double *datasetInput = (double *)malloc(neuronShape * datasetLines * sizeof(double));
    if (datasetInput == NULL) {
        printf("\n");
        return EOF;
    }
    double *datasetTarget = (double *)malloc(neuronShape * datasetLines * sizeof(double));
    if (datasetTarget == NULL) {
        printf("\n");
        return EOF;
    }
    CNeuron_Result readAttempt = cneuron_get_utf8_dataset(datasetFile, neuronShape, datasetInput, datasetTarget);
    if (readAttempt.code != 0) {
        printf("\n%s\n", readAttempt.error);
        return EOF;
    }

    // Display the loaded `UTF-8` Prompt Dataset
    printf("\n");
    printf("Split UTF-8 Dataset:");
    for (int i = 0; i < datasetLines; ++i) {
        printf("\n");
        for (int u = 0; u < neuronShape; ++u) {
            printf(" %.15f", datasetInput[(i * neuronShape) + u]);
        }
        printf(" |");
        for (int u = 0; u < neuronShape; ++u) {
            printf(" %.15f", datasetTarget[(i * neuronShape) + u]);
        }
    }

    // Train the CNeuron_Neuron
    int epochs = 40000;
    clock_t start, end;
    start = clock();
    CNeuron_Result neuronTrain = cneuron_neuron_train(&myNeuron, datasetInput, datasetTarget, datasetLines, epochs);
    if (neuronTrain.code == EOF) {
        printf("\n%s\n", neuronTrain.error);
        return EOF;
    }
    end = clock() - start;
    int time_taken = (int)end;
    printf("\nTraining took %d milliseconds to execute %d epochs of %d lines of size %d [%d bytes].\n", time_taken, epochs, datasetLines, myNeuron.shape, datasetLines * myNeuron.shape * 32);

    // Print the trained weights and bias
    printf("\nTrained Weights:");
    for (int i = 0; i < myNeuron.shape; ++i) {
        printf("\n\t%d: %.15f", i, myNeuron.weights[i]);
    }
    printf("\nTrained Bias:");
    for (int i = 0; i < myNeuron.shape; ++i) {
        printf("\n\t%d: %.15f", i, myNeuron.bias[i]);
    }

    // Test the CNeuron_Neuron
    double testIn[myNeuron.shape];
    double testOut[myNeuron.shape];
    printf("\nPrompt:");
    for (int i = 0; i < myNeuron.shape; ++i) {
        double inp = (double)'a' / UTF8_TKN_BASE;
        printf("\n\t%d: %.15f", i, inp);
        testIn[i] = inp;
    }
    CNeuron_Result neuronForward = cneuron_neuron_forward(&myNeuron, testIn, testOut);
    if (neuronForward.code == EOF) {
        printf("\n%s\n", neuronForward.error);
        return EOF;
    }
    printf("\nPrediction:");
    double avgAccuracy = 0.0;
    for (int i = 0; i < myNeuron.shape; ++i) {
        double x = (testOut[i] / 0.3828125) * 100.0;
        printf("\n\t%d: %.15f | 0.3828125 | %.3f", i, testOut[i], x);
        avgAccuracy += x;
    }
    avgAccuracy /= myNeuron.shape;
    printf("\nAverage Prediction Accuracy = %.15f\nAccuracy must be 99.60785 to 100.39215 for UTF-8 tokens.\n", avgAccuracy);
    return 0;
}

int main_test() {
    int t = test_function();
    char *func = (char *)__FUNCTION__;
    CNeuron_Result r =  cneuron_result(0, func, "OK");
    printf("\n[%d] %s\n", r.code, r.error);
    return t;
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        return main_test();
    }
    return main_test();
}
