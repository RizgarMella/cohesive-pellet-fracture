// Smoothed-aggregation algebraic multigrid (Vanek, Mandel & Brezina 1996)
// used as a PCG preconditioner. Elasticity uses the six rigid-body modes as
// near-nullspace, heat conduction the constant. Aggregation follows the
// strength of connection, so cracked (softened) interfaces automatically
// stop being aggregated across. Chebyshev smoothing and OpenMP-parallel
// SpGEMM keep every level thread-parallel; the coarsest level is solved by
// dense Cholesky.
#pragma once
#include <array>
#include <vector>

#include "sparse.hpp"

namespace pcz {

struct CSR {
  int nrow = 0, ncol = 0;
  std::vector<int> ptr, col;
  std::vector<double> val;
  void matvec(const double* x, double* y) const;
  CSR transpose() const;
};
CSR spgemm(const CSR& A, const CSR& B);

class AMG {
public:
  struct Params {
    double theta = 0.02;     // strength threshold
    int max_coarse = 1500;   // dofs on the coarsest level
    int max_levels = 12;
    int cheb_degree = 3;
    double cheb_ratio = 25;  // smoothing interval [lmax/ratio, 1.1 lmax]
  } prm;

  // A: block matrix (block size 1 or 3); mask: Dirichlet mask (may be empty);
  // coords: node coordinates (needed for rigid-body modes when B = 3)
  void setup(const BlockCSR& A, const std::vector<double>& mask, const std::vector<std::array<double, 3>>* coords);
  void apply(const std::vector<double>& r, std::vector<double>& z) const;  // z = M^-1 r (one V-cycle)
  bool ready() const { return !levels_.empty(); }
  int nlevels() const { return int(levels_.size()); }
  double operator_complexity() const;

private:
  struct Level {
    CSR A, P, R;
    std::vector<double> dinv;
    double lmax = 1;
    int bs = 1;  // dofs per node
  };
  std::vector<Level> levels_;
  std::vector<double> chol_;  // dense Cholesky of the coarsest operator
  int nc_ = 0;
  void vcycle(int l, const std::vector<double>& b, std::vector<double>& x) const;
  void smooth(const Level& L, const std::vector<double>& b, std::vector<double>& x, bool zero_guess) const;
};

}  // namespace pcz
