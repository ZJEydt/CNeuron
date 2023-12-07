# CNeuron:

## Basic Singular Neural Network Neuron Implementation in C

CNeuron is a simple singular neural network neuron implementation in C, designed for easy integration into C projects.

This implementation focuses on a single neuron with functions for initialization, training, saving, loading, and processing data.

The neuron utilizes Leaky Rectified Linear Unit (LeakyReLU) & Stochastic Gradient Descent with Momentum (SGDM).

**Please Note**: This was a side project built out of pure curiosity as I was learning the C programming language and is only intended for educational purposes.

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

### Contributing

Contributions are welcome! Please follow the Contribution Guidelines.

### License

This project is licensed under the **MIT License** - see the *LICENSE* file for details.

Make sure to replace placeholder sections with appropriate values for your project.
