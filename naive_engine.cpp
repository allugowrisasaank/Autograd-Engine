#include <vector>
#include <memory>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>
#include <set>

using f32 = float;
using NodePtr = std::shared_ptr<struct Node>;

struct Tensor {
    std::vector<f32> data;
    std::size_t rows, cols;
    Tensor(std::size_t r = 0, std::size_t c = 0, f32 val = 0.0f) : data(r * c, val), rows(r), cols(c) {}
};

struct Node {
    Tensor data;
    Tensor grad;
    bool requires_grad;
    NodePtr p0, p1;

    Node(Tensor d, bool req_grad = true, NodePtr a = nullptr, NodePtr b = nullptr) 
        : data(std::move(d)), grad(data.rows, data.cols, 0.0f), requires_grad(req_grad), p0(a), p1(b) {}
    
    virtual void backward_step() {}
    virtual ~Node() = default;
};

// Object-Oriented Operations (VTable Overhead)
struct AddBiasNode : public Node {
    AddBiasNode(NodePtr a, NodePtr b) : Node(Tensor(a->data.rows, a->data.cols), a->requires_grad || b->requires_grad, a, b) {
        for (std::size_t i = 0; i < a->data.rows; ++i)
            for (std::size_t j = 0; j < a->data.cols; ++j)
                data.data[i * a->data.cols + j] = a->data.data[i * a->data.cols + j] + b->data.data[j];
    }
    void backward_step() override {
        if (p0->requires_grad) {
            for (std::size_t i = 0; i < data.rows; ++i)
                for (std::size_t j = 0; j < data.cols; ++j)
                    p0->grad.data[i * data.cols + j] += grad.data[i * data.cols + j];
        }
        if (p1->requires_grad) {
            for (std::size_t i = 0; i < data.rows; ++i)
                for (std::size_t j = 0; j < data.cols; ++j)
                    p1->grad.data[j] += grad.data[i * data.cols + j];
        }
    }
};

struct MatMulNode : public Node {
    MatMulNode(NodePtr a, NodePtr b) : Node(Tensor(a->data.rows, b->data.cols), a->requires_grad || b->requires_grad, a, b) {
        // Naive O(N^3) without Cache Tiling/Blocking
        for (std::size_t i = 0; i < a->data.rows; ++i) {
            for (std::size_t j = 0; j < b->data.cols; ++j) {
                f32 sum = 0;
                for (std::size_t k = 0; k < a->data.cols; ++k) {
                    sum += a->data.data[i * a->data.cols + k] * b->data.data[k * b->data.cols + j];
                }
                data.data[i * data.cols + j] = sum;
            }
        }
    }
    void backward_step() override {
        if (p0->requires_grad) {
            for (std::size_t i = 0; i < p0->data.rows; ++i)
                for (std::size_t j = 0; j < p1->data.cols; ++j)
                    for (std::size_t k = 0; k < p0->data.cols; ++k)
                        p0->grad.data[i * p0->data.cols + k] += grad.data[i * data.cols + j] * p1->data.data[k * p1->data.cols + j];
        }
        if (p1->requires_grad) {
            for (std::size_t i = 0; i < p0->data.rows; ++i)
                for (std::size_t k = 0; k < p0->data.cols; ++k)
                    for (std::size_t j = 0; j < p1->data.cols; ++j)
                        p1->grad.data[k * p1->data.cols + j] += p0->data.data[i * p0->data.cols + k] * grad.data[i * data.cols + j];
        }
    }
};

struct ReluNode : public Node {
    ReluNode(NodePtr a) : Node(Tensor(a->data.rows, a->data.cols), a->requires_grad, a) {
        for (std::size_t i = 0; i < a->data.data.size(); ++i)
            data.data[i] = a->data.data[i] > 0 ? a->data.data[i] : 0;
    }
    void backward_step() override {
        if (p0->requires_grad) {
            for (std::size_t i = 0; i < p0->data.data.size(); ++i)
                if (p0->data.data[i] > 0) p0->grad.data[i] += grad.data[i];
        }
    }
};

struct MseNode : public Node {
    MseNode(NodePtr p, NodePtr t) : Node(Tensor(1, 1), p->requires_grad, p, t) {
        f32 sum = 0;
        for (std::size_t i = 0; i < p->data.data.size(); ++i) {
            f32 diff = p->data.data[i] - t->data.data[i];
            sum += diff * diff;
        }
        data.data[0] = sum / p->data.data.size();
    }
    void backward_step() override {
        if (p0->requires_grad) {
            f32 factor = 2.0f / p0->data.data.size();
            for (std::size_t i = 0; i < p0->data.data.size(); ++i)
                p0->grad.data[i] += factor * (p0->data.data[i] - p1->data.data[i]) * grad.data[0];
        }
    }
};

void backward(NodePtr root) {
    std::vector<NodePtr> topo;
    std::set<Node*> visited;
    auto dfs = [&](auto& self, NodePtr n) -> void {
        if (!n || visited.count(n.get())) return;
        visited.insert(n.get());
        self(self, n->p0);
        self(self, n->p1);
        topo.push_back(n);
    };
    dfs(dfs, root);
    
    root->grad.data[0] = 1.0f;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->backward_step(); // Virtual function call overhead!
    }
}

struct Linear {
    NodePtr W, b;
    Linear(std::size_t in, std::size_t out) {
        f32 limit = std::sqrt(6.0f / (in + out));
        std::mt19937 rng(42);
        std::uniform_real_distribution<f32> dist(-limit, limit);
        
        W = std::make_shared<Node>(Tensor(in, out), true);
        for (auto& val : W->data.data) val = dist(rng);
        
        b = std::make_shared<Node>(Tensor(1, out, 0.0f), true);
    }
    NodePtr operator()(NodePtr x) {
        // Massive std::make_shared overhead every epoch
        return std::make_shared<AddBiasNode>(std::make_shared<MatMulNode>(x, W), b);
    }
};

struct ScopedTimer {
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
    ScopedTimer(std::string n) : name(std::move(n)), start(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "[Profile] " << name << ": " << duration << " us\n";
    }
};

int main() {
    Linear l1(2, 16), l2(16, 16), l3(16, 1);
    
    Tensor x_data(4, 2);
    x_data.data = {0,0, 0,1, 1,0, 1,1};
    Tensor y_data(4, 1);
    y_data.data = {0, 1, 1, 0};
    
    std::cout << "Starting NAIVE Object-Oriented Autograd Engine...\n";
    std::cout << "(Using std::shared_ptr, dynamic memory, and virtual functions)\n";
    std::cout << "--------------------------------------------------------\n";
    {
        ScopedTimer t("Total Training Time (1000 Epochs)");
        for (int epoch = 0; epoch <= 1000; ++epoch) {
            std::fill(l1.W->grad.data.begin(), l1.W->grad.data.end(), 0.0f);
            std::fill(l1.b->grad.data.begin(), l1.b->grad.data.end(), 0.0f);
            std::fill(l2.W->grad.data.begin(), l2.W->grad.data.end(), 0.0f);
            std::fill(l2.b->grad.data.begin(), l2.b->grad.data.end(), 0.0f);
            std::fill(l3.W->grad.data.begin(), l3.W->grad.data.end(), 0.0f);
            std::fill(l3.b->grad.data.begin(), l3.b->grad.data.end(), 0.0f);
            
            // Forward pass - 11 dynamic heap allocations happen here per epoch!
            NodePtr x = std::make_shared<Node>(x_data, false);
            NodePtr y = std::make_shared<Node>(y_data, false);
            
            NodePtr x1 = l1(x);
            NodePtr a1 = std::make_shared<ReluNode>(x1);
            NodePtr x2 = l2(a1);
            NodePtr a2 = std::make_shared<ReluNode>(x2);
            NodePtr pred = l3(a2);
            NodePtr loss = std::make_shared<MseNode>(pred, y);
            
            // Backward pass - pointer chasing and virtual vtable lookups
            backward(loss);
            
            f32 lr = 0.05f;
            for(size_t i=0; i<l1.W->data.data.size(); ++i) l1.W->data.data[i] -= lr * l1.W->grad.data[i];
            for(size_t i=0; i<l1.b->data.data.size(); ++i) l1.b->data.data[i] -= lr * l1.b->grad.data[i];
            for(size_t i=0; i<l2.W->data.data.size(); ++i) l2.W->data.data[i] -= lr * l2.W->grad.data[i];
            for(size_t i=0; i<l2.b->data.data.size(); ++i) l2.b->data.data[i] -= lr * l2.b->grad.data[i];
            for(size_t i=0; i<l3.W->data.data.size(); ++i) l3.W->data.data[i] -= lr * l3.W->grad.data[i];
            for(size_t i=0; i<l3.b->data.data.size(); ++i) l3.b->data.data[i] -= lr * l3.b->grad.data[i];
            
            if (epoch % 100 == 0) std::cout << "Epoch " << epoch << " | Loss: " << loss->data.data[0] << "\n";
            
            // Scope ends: std::shared_ptr destructors fire, triggering 11 dynamic memory deallocations
        } 
    }
    return 0;
}
