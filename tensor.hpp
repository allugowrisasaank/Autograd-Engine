#pragma once
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ag {

using f32 = float;
using u32 = std::uint32_t;
using i32 = std::int32_t;

class Tensor {
public:
  using Storage = std::vector<f32>;

private:
  Storage data_;
  std::size_t rows_;
  std::size_t cols_;

public:
  Tensor() : rows_(0), cols_(0) {}
  Tensor(std::size_t rows, std::size_t cols, f32 init_val = 0.0f)
      : data_(rows * cols, init_val), rows_(rows), cols_(cols) {}

  [[nodiscard]] std::size_t rows() const { return rows_; }
  [[nodiscard]] std::size_t cols() const { return cols_; }
  [[nodiscard]] std::size_t size() const { return data_.size(); }
  [[nodiscard]] const f32 *data() const { return data_.data(); }
  [[nodiscard]] f32 *data() { return data_.data(); }

  [[nodiscard]] f32 &operator()(std::size_t r, std::size_t c) {
    if (r >= rows_ || c >= cols_)
      throw std::out_of_range("Tensor index out of bounds");
    return data_[r * cols_ + c];
  }

  [[nodiscard]] const f32 &operator()(std::size_t r, std::size_t c) const {
    if (r >= rows_ || c >= cols_)
      throw std::out_of_range("Tensor index out of bounds");
    return data_[r * cols_ + c];
  }

  [[nodiscard]] f32 &operator[](std::size_t idx) {
    if (idx >= data_.size())
      throw std::out_of_range("Tensor index out of bounds");
    return data_[idx];
  }

  [[nodiscard]] const f32 &operator[](std::size_t idx) const {
    if (idx >= data_.size())
      throw std::out_of_range("Tensor index out of bounds");
    return data_[idx];
  }

  // Fill with value
  void fill(f32 val) { std::fill(data_.begin(), data_.end(), val); }
};

// Math Operations
[[nodiscard]] inline Tensor add(const Tensor &a, const Tensor &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols())
    throw std::invalid_argument("Shape mismatch in add");
  Tensor res(a.rows(), a.cols());
  const f32 *a_ptr = a.data();
  const f32 *b_ptr = b.data();
  f32 *res_ptr = res.data();
  std::size_t sz = a.size();
  for (std::size_t i = 0; i < sz; ++i) {
    res_ptr[i] = a_ptr[i] + b_ptr[i];
  }
  return res;
}

[[nodiscard]] inline Tensor multiply(const Tensor &a, const Tensor &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols())
    throw std::invalid_argument("Shape mismatch in multiply");
  Tensor res(a.rows(), a.cols());
  const f32 *a_ptr = a.data();
  const f32 *b_ptr = b.data();
  f32 *res_ptr = res.data();
  std::size_t sz = a.size();
  for (std::size_t i = 0; i < sz; ++i) {
    res_ptr[i] = a_ptr[i] * b_ptr[i];
  }
  return res;
}

[[nodiscard]] inline Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.cols() != b.rows())
    throw std::invalid_argument("Shape mismatch in matmul");
  Tensor res(a.rows(), b.cols());

  const f32 *a_ptr = a.data();
  const f32 *b_ptr = b.data();
  f32 *res_ptr = res.data();

  std::size_t M = a.rows();
  std::size_t K = a.cols();
  std::size_t N = b.cols();

  // Cache-friendly blocked matrix multiplication
  constexpr std::size_t BLOCK_SIZE = 32;
  for (std::size_t i = 0; i < M; i += BLOCK_SIZE) {
    for (std::size_t k = 0; k < K; k += BLOCK_SIZE) {
      for (std::size_t j = 0; j < N; j += BLOCK_SIZE) {
        std::size_t i_end = std::min(i + BLOCK_SIZE, M);
        std::size_t k_end = std::min(k + BLOCK_SIZE, K);
        std::size_t j_end = std::min(j + BLOCK_SIZE, N);

        for (std::size_t ii = i; ii < i_end; ++ii) {
          for (std::size_t kk = k; kk < k_end; ++kk) {
            f32 a_val = a_ptr[ii * K + kk];
            for (std::size_t jj = j; jj < j_end; ++jj) {
              res_ptr[ii * N + jj] += a_val * b_ptr[kk * N + jj];
            }
          }
        }
      }
    }
  }
  return res;
}

} // namespace ag
