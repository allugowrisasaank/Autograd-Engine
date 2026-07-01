#pragma once
#include "tensor.hpp"
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ag {

enum class OpType { Leaf, Add, AddBias, Multiply, MatMul, ReLU, MSE };

using NodeIndex = u32;
constexpr NodeIndex NULL_NODE = static_cast<NodeIndex>(-1);

struct Node {
  Tensor data;
  Tensor grad;
  OpType op;
  NodeIndex parent0;
  NodeIndex parent1;
  bool requires_grad;

  Node()
      : data(), grad(), op(OpType::Leaf), parent0(NULL_NODE),
        parent1(NULL_NODE), requires_grad(false) {}

  Node(Tensor d, OpType o, NodeIndex p0 = NULL_NODE, NodeIndex p1 = NULL_NODE,
       bool req_grad = true)
      : data(std::move(d)), grad(data.rows(), data.cols(), 0.0f), op(o),
        parent0(p0), parent1(p1), requires_grad(req_grad) {}
};

class Graph {
private:
  std::vector<Node> nodes_;
  std::size_t param_count_ = 0;

public:
  Graph() {
    nodes_.reserve(100000); // Preallocate to avoid reallocations
  }

  [[nodiscard]] NodeIndex create_leaf(Tensor data, bool requires_grad = true) {
    nodes_.emplace_back(std::move(data), OpType::Leaf, NULL_NODE, NULL_NODE,
                        requires_grad);
    return static_cast<NodeIndex>(nodes_.size() - 1);
  }

  [[nodiscard]] NodeIndex add(NodeIndex a, NodeIndex b) {
    const Node &na = nodes_[a];
    const Node &nb = nodes_[b];
    nodes_.emplace_back(ag::add(na.data, nb.data), OpType::Add, a, b,
                        na.requires_grad || nb.requires_grad);
    return static_cast<NodeIndex>(nodes_.size() - 1);
  }

  [[nodiscard]] NodeIndex add_bias(NodeIndex a, NodeIndex b) {
    const Node &na = nodes_[a];
    const Node &nb = nodes_[b];
    if (nb.data.rows() != 1 || nb.data.cols() != na.data.cols()) {
      throw std::invalid_argument("Shape mismatch in add_bias");
    }

    Tensor res(na.data.rows(), na.data.cols());
    const f32 *a_ptr = na.data.data();
    const f32 *b_ptr = nb.data.data();
    f32 *res_ptr = res.data();
    std::size_t M = na.data.rows();
    std::size_t N = na.data.cols();

    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        res_ptr[i * N + j] = a_ptr[i * N + j] + b_ptr[j];
      }
    }

    nodes_.emplace_back(std::move(res), OpType::AddBias, a, b,
                        na.requires_grad || nb.requires_grad);
    return static_cast<NodeIndex>(nodes_.size() - 1);
  }

  [[nodiscard]] NodeIndex multiply(NodeIndex a, NodeIndex b) {
    const Node &na = nodes_[a];
    const Node &nb = nodes_[b];
    nodes_.emplace_back(ag::multiply(na.data, nb.data), OpType::Multiply, a, b,
                        na.requires_grad || nb.requires_grad);
    return static_cast<NodeIndex>(nodes_.size() - 1);
  }

  [[nodiscard]] NodeIndex matmul(NodeIndex a, NodeIndex b) {
    const Node &na = nodes_[a];
    const Node &nb = nodes_[b];
    nodes_.emplace_back(ag::matmul(na.data, nb.data), OpType::MatMul, a, b,
                        na.requires_grad || nb.requires_grad);
    return static_cast<NodeIndex>(nodes_.size() - 1);
  }

  [[nodiscard]] NodeIndex relu(NodeIndex a) {
    const Node &na = nodes_[a];
    Tensor res(na.data.rows(), na.data.cols());
    const f32 *a_ptr = na.data.data();
    f32 *res_ptr = res.data();
    std::size_t sz = na.data.size();
    for (std::size_t i = 0; i < sz; ++i) {
      res_ptr[i] = a_ptr[i] > 0.0f ? a_ptr[i] : 0.0f;
    }
    nodes_.emplace_back(std::move(res), OpType::ReLU, a, NULL_NODE,
                        na.requires_grad);
    return static_cast<NodeIndex>(nodes_.size() - 1);
  }

  [[nodiscard]] NodeIndex mse(NodeIndex pred, NodeIndex target) {
    const Node &np = nodes_[pred];
    const Node &nt = nodes_[target];
    if (np.data.size() != nt.data.size())
      throw std::invalid_argument("Shape mismatch in MSE");

    f32 sum_sq = 0.0f;
    const f32 *p_ptr = np.data.data();
    const f32 *t_ptr = nt.data.data();
    std::size_t sz = np.data.size();
    for (std::size_t i = 0; i < sz; ++i) {
      f32 diff = p_ptr[i] - t_ptr[i];
      sum_sq += diff * diff;
    }
    f32 mse_val = sum_sq / static_cast<f32>(sz);
    Tensor res(1, 1, mse_val);
    nodes_.emplace_back(std::move(res), OpType::MSE, pred, target,
                        np.requires_grad);
    return static_cast<NodeIndex>(nodes_.size() - 1);
  }

  void backward(NodeIndex root) {
    std::vector<NodeIndex> topo;
    std::vector<bool> visited(nodes_.size(), false);

    // Lambda for DFS
    auto dfs = [&](auto &self, NodeIndex idx) -> void {
      if (idx == NULL_NODE || visited[idx])
        return;
      visited[idx] = true;
      const Node &n = nodes_[idx];
      if (n.parent0 != NULL_NODE)
        self(self, n.parent0);
      if (n.parent1 != NULL_NODE)
        self(self, n.parent1);
      topo.push_back(idx);
    };

    dfs(dfs, root);

    // Seed gradient for root
    nodes_[root].grad.fill(1.0f);

    // Backward pass
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      NodeIndex idx = *it;
      Node &n = nodes_[idx];
      if (!n.requires_grad)
        continue;

      const f32 *g_ptr = n.grad.data();

      switch (n.op) {
      case OpType::Add: {
        Node &p0 = nodes_[n.parent0];
        Node &p1 = nodes_[n.parent1];
        f32 *g0_ptr = p0.grad.data();
        f32 *g1_ptr = p1.grad.data();
        std::size_t sz = n.grad.size();
        for (std::size_t i = 0; i < sz; ++i) {
          if (p0.requires_grad)
            g0_ptr[i] += g_ptr[i];
          if (p1.requires_grad)
            g1_ptr[i] += g_ptr[i];
        }
        break;
      }
      case OpType::AddBias: {
        Node &p0 = nodes_[n.parent0];
        Node &p1 = nodes_[n.parent1];
        f32 *g0_ptr = p0.grad.data();
        f32 *g1_ptr = p1.grad.data();
        std::size_t M = p0.data.rows();
        std::size_t N = p0.data.cols();
        for (std::size_t i = 0; i < M; ++i) {
          for (std::size_t j = 0; j < N; ++j) {
            if (p0.requires_grad)
              g0_ptr[i * N + j] += g_ptr[i * N + j];
            if (p1.requires_grad)
              g1_ptr[j] += g_ptr[i * N + j];
          }
        }
        break;
      }
      case OpType::Multiply: {
        Node &p0 = nodes_[n.parent0];
        Node &p1 = nodes_[n.parent1];
        const f32 *d0_ptr = p0.data.data();
        const f32 *d1_ptr = p1.data.data();
        f32 *g0_ptr = p0.grad.data();
        f32 *g1_ptr = p1.grad.data();
        std::size_t sz = n.grad.size();
        for (std::size_t i = 0; i < sz; ++i) {
          if (p0.requires_grad)
            g0_ptr[i] += g_ptr[i] * d1_ptr[i];
          if (p1.requires_grad)
            g1_ptr[i] += g_ptr[i] * d0_ptr[i];
        }
        break;
      }
      case OpType::MatMul: {
        Node &p0 = nodes_[n.parent0];
        Node &p1 = nodes_[n.parent1];

        if (p0.requires_grad) {
          std::size_t M = p0.data.rows();
          std::size_t K = p0.data.cols();
          std::size_t N = p1.data.cols();
          f32 *g0_ptr = p0.grad.data();
          const f32 *d1_ptr = p1.data.data();

          for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
              f32 grad_val = g_ptr[i * N + j];
              for (std::size_t k = 0; k < K; ++k) {
                g0_ptr[i * K + k] += grad_val * d1_ptr[k * N + j];
              }
            }
          }
        }
        if (p1.requires_grad) {
          std::size_t M = p0.data.rows();
          std::size_t K = p0.data.cols();
          std::size_t N = p1.data.cols();
          f32 *g1_ptr = p1.grad.data();
          const f32 *d0_ptr = p0.data.data();

          for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t k = 0; k < K; ++k) {
              f32 a_val = d0_ptr[i * K + k];
              for (std::size_t j = 0; j < N; ++j) {
                g1_ptr[k * N + j] += a_val * g_ptr[i * N + j];
              }
            }
          }
        }
        break;
      }
      case OpType::ReLU: {
        Node &p0 = nodes_[n.parent0];
        if (p0.requires_grad) {
          const f32 *d0_ptr = p0.data.data();
          f32 *g0_ptr = p0.grad.data();
          std::size_t sz = n.grad.size();
          for (std::size_t i = 0; i < sz; ++i) {
            if (d0_ptr[i] > 0.0f) {
              g0_ptr[i] += g_ptr[i];
            }
          }
        }
        break;
      }
      case OpType::MSE: {
        Node &pred = nodes_[n.parent0];
        if (pred.requires_grad) {
          Node &target = nodes_[n.parent1];
          const f32 *p_ptr = pred.data.data();
          const f32 *t_ptr = target.data.data();
          f32 *g0_ptr = pred.grad.data();
          std::size_t sz = pred.data.size();
          f32 factor = 2.0f / static_cast<f32>(sz);
          f32 g_val = g_ptr[0];

          for (std::size_t i = 0; i < sz; ++i) {
            g0_ptr[i] += factor * (p_ptr[i] - t_ptr[i]) * g_val;
          }
        }
        break;
      }
      case OpType::Leaf:
        break;
      }
    }
  }

  void step(f32 lr) {
    for (std::size_t i = 0; i < param_count_; ++i) {
      if (nodes_[i].requires_grad) {
        f32 *d_ptr = nodes_[i].data.data();
        const f32 *g_ptr = nodes_[i].grad.data();
        std::size_t sz = nodes_[i].data.size();
        for (std::size_t j = 0; j < sz; ++j) {
          d_ptr[j] -= lr * g_ptr[j];
        }
      }
    }
  }

  void zero_grad() {
    for (auto &n : nodes_) {
      n.grad.fill(0.0f);
    }
  }

  Node &get_node(NodeIndex idx) { return nodes_[idx]; }

  void freeze_parameters() { param_count_ = nodes_.size(); }

  void reset_computations() {
    nodes_.erase(nodes_.begin() + param_count_, nodes_.end());
  }

  [[nodiscard]] std::size_t memory_used() const {
    return nodes_.capacity() * sizeof(Node) +
           nodes_.capacity() * nodes_[0].data.size() * sizeof(f32) * 2;
  }

  [[nodiscard]] std::size_t node_count() const { return nodes_.size(); }
};

} // namespace ag
