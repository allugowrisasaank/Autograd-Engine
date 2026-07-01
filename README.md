# C++ Autograd Engine

A highly optimized, zero-dependency Automatic Differentiation (Autograd) engine and Neural Network framework built entirely from scratch in C++20.

## Overview
This project is an educational, performance-focused machine learning engine that implements its own Tensor math operations, a backward-pass topological sort (backpropagation), and Multi-Layer Perceptrons (MLPs). It proves that you can achieve blazingly fast ML training cycles without massive frameworks like PyTorch or heavy math libraries like Eigen/BLAS.

### Core Architectural Features
* **Zero External Dependencies:** Built purely with the C++20 Standard Template Library (STL).
* **Arena Memory Allocator:** Makes use of an Object Pool (`std::vector<Node>`) for the computation graph. This completely eliminates dynamic heap allocations during the forward and backward passes.
* **Cache-Locality:** Tensors utilize contiguous arrays for high CPU cache hit rates, and matrix multiplication is implemented using cache-line blocking (tiling).
* **Strict Type Safety:** Avoids implicit conversions and uses native types (`std::size_t`, `f32`, `u32`).

## Project Structure
* `tensor.hpp`: The core multidimensional array struct, housing basic vector arithmetic and blocked matrix multiplication.
* `autograd.hpp`: The directed acyclic graph (DAG) engine. It records operations, executes DFS topological sorting, calculates analytic derivatives, and manages the zero-allocation memory pool.
* `nn.hpp`: Contains high-level Machine Learning constructs like `Linear` layers, `ReLU`, and `MSE` Loss, combining them into an `MLP`.
* `main.cpp`: The entry point and validation file. Sets up a 3-layer neural network, trains it on a synthesized non-linear XOR dataset using Stochastic Gradient Descent (SGD), and embeds microsecond-resolution profiling.

## Compilation & Execution

To compile the engine, ensure you have a compiler that supports C++20 (like GCC 11+ or Clang 13+). 

Run the following command in your terminal:
```bash
g++ -std=c++20 -O3 -march=native -Wall -Wextra -Werror main.cpp -o autograd_engine
```

**Optimization Flags Used:**
* `-O3`: Applies maximum algorithmic and memory optimizations.
* `-march=native`: Unlocks your CPU's specific vectorization instructions (e.g., AVX2/AVX-512).

Once compiled, execute the binary:
```bash
./autograd_engine
```

## Expected Output
The engine will begin training a 3-layer MLP on the XOR problem for 1000 epochs. It will output loss metrics and memory telemetry, completing the entire training loop in under 3 milliseconds:

```text
Starting C++ Autograd Engine Training Loop...
Optimizing XOR Dataset with 3-Layer MLP
--------------------------------------------------------
Epoch    0 | Loss: 0.511369 | Active Nodes: 17
Epoch  100 | Loss: 0.002962 | Active Nodes: 17
Epoch  200 | Loss: 0.000008 | Active Nodes: 17
...
Epoch 1000 | Loss: 0.000000 | Active Nodes: 17
[Profile] Total Training Time (1000 Epochs): 2182 us
--------------------------------------------------------
Final Predictions for XOR Dataset:
0.000000 XOR 0.000000 = 0.0000  (Predicted: 0)
0.0000 XOR 1.0000 = 1.0000  (Predicted: 1)
1.0000 XOR 0.0000 = 1.0000  (Predicted: 1)
1.0000 XOR 1.0000 = 0.0000  (Predicted: 0)
```
*(Notice the `Active Nodes` count stays locked at 17 across epochs, validating the zero-allocation arena design).*
