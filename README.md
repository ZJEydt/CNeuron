# CNeuron: Basic Singular Neural Network Neuron Implementation in C

CNeuron is a simple singular neural network neuron implementation in C, designed for easy integration into C projects.

This implementation focuses on a single neuron with functions for initialization, training, saving, loading, and processing data.

```
**PLEASE NOTE**
This implementation utilizes Leaky Rectified Linear Unit (LeakyReLU) & Stochastic Gradient Descent with Momentum (SGDM)
```

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Neuron Initialization](#neuron-initialization)
  - [Forward Propagation](#forward-propagation)
  - [Training](#training)
  - [Saving and Loading](#saving-and-loading)
  - [Dataset Processing](#dataset-processing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Neuron Initialization:** Initialize a neural network neuron with specified parameters.
- **Forward Propagation:** Perform forward propagation for a given input.
- **Training:** Train the neuron using backpropagation with customizable epochs and learning parameters.
- **Saving and Loading:** Save and load neuron parameters to/from a file.
- **Dataset Processing:** Read datasets from files in UTF-8 format.

## Getting Started

### Prerequisites

- C compiler (e.g., GCC)
- Standard C libraries (math.h, stdio.h, stdlib.h, string.h, time.h)

### Installation

#### Clone the repository:

```bash
git clone https://github.com/ZJEydt/NeuralC.git
cd NeuralC
```

#### Compile the test code:

```bash
gcc NeuralC_Test.c NeuralC_Neuron.c -o NeuralC_Test
```

### Usage

#### Neuron Initialization

```c
#include "neuralc.h"

NeuralC_Neuron neuron;
NeuralC_Result result = neuralc_neuron_init(&neuron, "example", 10, 0.01, 0.1, 0.1, 0.9);

if (result.code == 0) {
    // Neuron initialization successful
} else {
    // Handle initialization error
    printf("Error: %s\n", result.error);
}
```

#### Forward Propagation

```c
double input[10] = { /* Your input values */ };
double output[10];

NeuralC_Result result = neuralc_neuron_forward(&neuron, input, output);

if (result.code == 0) {
    // Forward propagation successful
} else {
    // Handle forward propagation error
    printf("Error: %s\n", result.error);
}
```

#### Training

```c
double inputs[/* Number of Training Samples */][10] = { /* Your input values */ };
double targets[/* Number of Training Samples */][10] = { /* Your target values */ };

NeuralC_Result result = neuralc_neuron_train(&neuron, inputs, targets, /* Number of Training Samples */, /* Number of Epochs */);

if (result.code == 0) {
    // Training successful
} else {
    // Handle training error
    printf("Error: %s\n", result.error);
}
```

#### Saving and Loading

```c
NeuralC_Result result = neuralc_neuron_save(&neuron, "neuron_params.txt");

if (result.code == 0) {
    // Saving successful
} else {
    // Handle saving error
    printf("Error: %s\n", result.error);
}
```

```c
NeuralC_Result result = neuralc_neuron_load(&neuron, "neuron_params.txt", "example");

if (result.code == 0) {
    // Loading successful
} else {
    // Handle loading error
    printf("Error: %s\n", result.error);
}
```

#### Dataset Processing

```c
double inputs[/* Number of Samples */][/* Shape */];
double targets[/* Number of Samples */][/* Shape */];

NeuralC_Result result = neuralc_get_utf8_dataset("your_dataset.txt", /* Shape */, inputs, targets);

if (result.code == 0) {
    // Dataset processing successful
} else {
    // Handle dataset processing error
    printf("Error: %s\n", result.error);
}
```

### Contributing

Contributions are welcome! Please follow the Contribution Guidelines.

### License

This project is licensed under the **MIT License** - see the *LICENSE* file for details.

Make sure to replace placeholder sections with appropriate values for your project.
