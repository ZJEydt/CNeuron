
#pragma once

#ifndef _CNEURON_NEURON_H_
    #define _CNEURON_NEURON_H_

    struct CNeuron_Result;

    CNeuron_Result cneuron_result(int code, char *func, char *error);

    struct CNeuron_Neuron;

    CNeuron_Result cneuron_neuron_init(CNeuron_Neuron *n, char *idx, int shape, double lrFactor, double nSlope, double velocity, double momentum);
    CNeuron_Result cneuron_neuron_save(CNeuron_Neuron *n, const char *f);
    CNeuron_Result cneuron_neuron_load(CNeuron_Neuron *n, char *f, char *neuronName);
    CNeuron_Result cneuron_neuron_forward(CNeuron_Neuron *n, double *in, double *out);
    CNeuron_Result cneuron_neuron_train(CNeuron_Neuron *n, double *inputs, double *targets, int lines, int epochs);

    int cneuron_count_lines_in_file(const char *f);
    CNeuron_Result neuralc_get_utf8_dataset(const char *f, const int shape, double *input, double *target);

#endif
