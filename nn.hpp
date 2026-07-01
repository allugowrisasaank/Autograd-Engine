#pragma once
#include "autograd.hpp"
#include <random>

namespace ag {

struct Linear {
  NodeIndex W;
  NodeIndex b;

  Linear(Graph &g, std::size_t in_features, std::size_t out_features) {
    f32 limit = std::sqrt(6.0f / static_cast<f32>(in_features + out_features));
    // Use random_device or fixed seed
    std::mt19937 rng(42);
    std::uniform_real_distribution<f32> dist(-limit, limit);

    Tensor w_data(in_features, out_features);
    for (std::size_t i = 0; i < w_data.size(); ++i)
      w_data[i] = dist(rng);
    W = g.create_leaf(std::move(w_data), true);

    Tensor b_data(1, out_features, 0.0f);
    b = g.create_leaf(std::move(b_data), true);
  }

  NodeIndex operator()(Graph &g, NodeIndex x) {
    NodeIndex mul = g.matmul(x, W);
    return g.add_bias(mul, b);
  }
};

struct MLP {
  Linear l1;
  Linear l2;
  Linear l3;

  MLP(Graph &g) : l1(g, 2, 16), l2(g, 16, 16), l3(g, 16, 1) {}

  NodeIndex operator()(Graph &g, NodeIndex x) {
    NodeIndex x1 = l1(g, x);
    NodeIndex a1 = g.relu(x1);
    NodeIndex x2 = l2(g, a1);
    NodeIndex a2 = g.relu(x2);
    NodeIndex x3 = l3(g, a2);
    return x3;
  }
};

} // namespace ag
