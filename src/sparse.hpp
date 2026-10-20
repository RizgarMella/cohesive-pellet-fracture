// Block-CSR sparse matrices sharing the mesh node graph, OpenMP-parallel
// mat-vec and a preconditioned conjugate gradient solver (block Jacobi).
// Dirichlet constraints are imposed by a mask (1 = free, 0 = fixed): the
// solver works on the masked operator  M A M + (I - M), which equals the
// classic row/column elimination without modifying the stored matrix.
#pragma once
#include <vector>

namespace pcz {

class AMG;

struct BlockCSR {
  int B = 1;      // block size (1 thermal, 3 mechanics)
  int n = 0;      // block rows
  const std::vector<int>* ptr = nullptr;
  const std::vector<int>* col = nullptr;
  std::vector<double> val;   // nnzb * B * B, row-major blocks
  std::vector<int> diag;     // index of the diagonal block in each row

  void init(int B_, const std::vector<int>& p, const std::vector<int>& c);
  void zero();
  int find(int row, int c) const;  // block index or -1
  double* block(int k) { return &val[size_t(k) * B * B]; }
  void matvec(const std::vector<double>& x, std::vector<double>& y) const;
  void add_diagonal(double s);  // A += s*I
};

struct PCGResult {
  int iters = 0;
  double rel_res = 0;
  bool converged = false;
};

// Solve A x = b. `mask` may be empty (no constraints). x holds the initial guess.
// M: optional AMG preconditioner (built on the same masked operator); block Jacobi otherwise.
PCGResult pcg(const BlockCSR& A, const std::vector<double>& b, std::vector<double>& x, const std::vector<double>& mask,
              double rtol, int maxit, const AMG* M = nullptr);

// small parallel vector helpers
double dot(const std::vector<double>& a, const std::vector<double>& b);
double norm2(const std::vector<double>& a);
double norm_inf(const std::vector<double>& a);

}  // namespace pcz
