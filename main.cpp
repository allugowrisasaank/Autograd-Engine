#include "autograd.hpp"
#include "nn.hpp"
#include "tensor.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

struct ScopedTimer {
  std::string name;
  std::chrono::high_resolution_clock::time_point start;
  ScopedTimer(std::string n)
      : name(std::move(n)), start(std::chrono::high_resolution_clock::now()) {}
  ~ScopedTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "[Profile] " << name << ": " << duration << " us\n";
  }
};

int main() {
  ag::Graph g;
  ag::MLP model(g);

  // Lock the parameter nodes in the arena so they persist
  g.freeze_parameters();

  // XOR Dataset Setup
  ag::Tensor x_data(4ULL, 2ULL);
  x_data(0, 0) = 0;
  x_data(0, 1) = 0;
  x_data(1, 0) = 0;
  x_data(1, 1) = 1;
  x_data(2, 0) = 1;
  x_data(2, 1) = 0;
  x_data(3, 0) = 1;
  x_data(3, 1) = 1;

  ag::Tensor y_data(4ULL, 1ULL);
  y_data(0, 0) = 0;
  y_data(1, 0) = 1;
  y_data(2, 0) = 1;
  y_data(3, 0) = 0;

  constexpr std::size_t EPOCHS = 1000;
  constexpr ag::f32 LR = 0.05f;

  std::cout << "Starting C++ Autograd Engine Training Loop...\n";
  std::cout << "Optimizing XOR Dataset with 3-Layer MLP\n";
  std::cout << "--------------------------------------------------------\n";
  {
    ScopedTimer t("Total Training Time (1000 Epochs)");
    for (std::size_t epoch = 0; epoch <= EPOCHS; ++epoch) {
      // Reset arena to only parameters (zero allocations per pass!)
      g.reset_computations();
      g.zero_grad();

      // Forward Pass
      ag::NodeIndex x = g.create_leaf(x_data, false);
      ag::NodeIndex y = g.create_leaf(y_data, false);
      ag::NodeIndex pred = model(g, x);
      ag::NodeIndex loss = g.mse(pred, y);

      // Backward Pass
      g.backward(loss);

      // Step (SGD)
      g.step(LR);

      if (epoch % 100 == 0) {
        ag::f32 current_loss = g.get_node(loss).data(0, 0);
        std::cout << "Epoch " << std::setw(4) << epoch
                  << " | Loss: " << std::fixed << std::setprecision(6)
                  << current_loss << " | Active Nodes: " << g.node_count()
                  << "\n";
      }
    }
  }

  std::cout << "--------------------------------------------------------\n";
  std::cout << "Final Predictions for XOR Dataset:\n";
  g.reset_computations();
  ag::NodeIndex x_test = g.create_leaf(x_data, false);
  ag::NodeIndex pred_test = model(g, x_test);
  const ag::Tensor &out = g.get_node(pred_test).data;

  for (std::size_t i = 0; i < out.rows(); ++i) {
    std::cout << x_data(i, 0) << " XOR " << x_data(i, 1) << " = " << std::fixed
              << std::setprecision(4) << out(i, 0);
    // Interpret output
    std::cout << "  (Predicted: " << (out(i, 0) > 0.5f ? 1 : 0) << ")\n";
  }

  std::cout << "\n[Metrics] Memory pre-allocation active nodes: "
            << g.node_count() << "\n";
  return 0;
}
