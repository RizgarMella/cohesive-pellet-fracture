#include "amg.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pcz {

void CSR::matvec(const double* x, double* y) const {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nrow; ++i) {
    double s = 0;
    for (int k = ptr[i]; k < ptr[i + 1]; ++k) s += val[k] * x[col[k]];
    y[i] = s;
  }
}

CSR CSR::transpose() const {
  CSR T;
  T.nrow = ncol;
  T.ncol = nrow;
  T.ptr.assign(ncol + 1, 0);
  for (int c : col) T.ptr[c + 1]++;
  for (int i = 0; i < ncol; ++i) T.ptr[i + 1] += T.ptr[i];
  T.col.resize(col.size());
  T.val.resize(val.size());
  std::vector<int> pos(T.ptr.begin(), T.ptr.end() - 1);
  for (int i = 0; i < nrow; ++i)
    for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
      int p = pos[col[k]]++;
      T.col[p] = i;
      T.val[p] = val[k];
    }
  return T;
}

CSR spgemm(const CSR& A, const CSR& B) {
  CSR C;
  C.nrow = A.nrow;
  C.ncol = B.ncol;
  C.ptr.assign(A.nrow + 1, 0);
#pragma omp parallel
  {
    std::vector<int> mark(B.ncol, -1);
#pragma omp for schedule(dynamic, 256)
    for (int i = 0; i < A.nrow; ++i) {
      int cnt = 0;
      for (int ka = A.ptr[i]; ka < A.ptr[i + 1]; ++ka) {
        int j = A.col[ka];
        for (int kb = B.ptr[j]; kb < B.ptr[j + 1]; ++kb)
          if (mark[B.col[kb]] != i) {
            mark[B.col[kb]] = i;
            ++cnt;
          }
      }
      C.ptr[i + 1] = cnt;
    }
  }
  for (int i = 0; i < A.nrow; ++i) C.ptr[i + 1] += C.ptr[i];
  C.col.resize(C.ptr[A.nrow]);
  C.val.resize(C.ptr[A.nrow]);
#pragma omp parallel
  {
    std::vector<int> mark(B.ncol, -1);
    std::vector<double> acc(B.ncol, 0.0);
#pragma omp for schedule(dynamic, 256)
    for (int i = 0; i < A.nrow; ++i) {
      int p0 = C.ptr[i], p = p0;
      for (int ka = A.ptr[i]; ka < A.ptr[i + 1]; ++ka) {
        int j = A.col[ka];
        double a = A.val[ka];
        for (int kb = B.ptr[j]; kb < B.ptr[j + 1]; ++kb) {
          int c = B.col[kb];
          if (mark[c] != i) {
            mark[c] = i;
            C.col[p++] = c;
            acc[c] = a * B.val[kb];
          } else {
            acc[c] += a * B.val[kb];
          }
        }
      }
      std::sort(C.col.begin() + p0, C.col.begin() + p);
      for (int k = p0; k < p; ++k) C.val[k] = acc[C.col[k]];
    }
  }
  return C;
}

namespace {

double lambda_max_DinvA(const CSR& A, const std::vector<double>& dinv) {
  const int n = A.nrow;
  std::vector<double> x(n), y(n);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1, 1);
  for (auto& v : x) v = U(rng);
  double lam = 1.0;
  for (int it = 0; it < 15; ++it) {
    double nx = 0;
    for (double v : x) nx += v * v;
    nx = std::sqrt(nx);
    if (nx == 0) return 1.0;
    for (auto& v : x) v /= nx;
    A.matvec(x.data(), y.data());
    double ny = 0;
    for (int i = 0; i < n; ++i) {
      y[i] *= dinv[i];
      ny += y[i] * y[i];
    }
    lam = std::sqrt(ny);
    std::swap(x, y);
  }
  return lam;
}

std::vector<double> inv_diag(const CSR& A) {
  std::vector<double> d(A.nrow, 1.0);
#pragma omp parallel for
  for (int i = 0; i < A.nrow; ++i)
    for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k)
      if (A.col[k] == i) d[i] = A.val[k] != 0.0 ? 1.0 / A.val[k] : 1.0;
  return d;
}

// ensure every row has a positive diagonal (rank-deficient coarse dofs become identity rows)
void fix_diagonal(CSR& A) {
  double dmax = 0;
  for (int i = 0; i < A.nrow; ++i)
    for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k)
      if (A.col[k] == i) dmax = std::max(dmax, std::abs(A.val[k]));
  const double tiny = 1e-13 * dmax;
  std::vector<int> missing;
  for (int i = 0; i < A.nrow; ++i) {
    bool found = false;
    for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k)
      if (A.col[k] == i) {
        found = true;
        if (std::abs(A.val[k]) <= tiny) {
          // isolate this dof
          for (int kk = A.ptr[i]; kk < A.ptr[i + 1]; ++kk) A.val[kk] = 0.0;
          A.val[k] = dmax > 0 ? dmax : 1.0;
        }
      }
    if (!found) missing.push_back(i);
  }
  if (missing.empty()) return;
  CSR B;
  B.nrow = A.nrow;
  B.ncol = A.ncol;
  B.ptr.assign(A.nrow + 1, 0);
  std::vector<char> miss(A.nrow, 0);
  for (int i : missing) miss[i] = 1;
  for (int i = 0; i < A.nrow; ++i) B.ptr[i + 1] = B.ptr[i] + (A.ptr[i + 1] - A.ptr[i]) + miss[i];
  B.col.resize(B.ptr.back());
  B.val.resize(B.ptr.back());
  for (int i = 0; i < A.nrow; ++i) {
    int p = B.ptr[i];
    bool ins = !miss[i];
    for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
      if (!ins && A.col[k] > i) {
        B.col[p] = i;
        B.val[p++] = dmax > 0 ? dmax : 1.0;
        ins = true;
      }
      B.col[p] = A.col[k];
      B.val[p++] = miss[i] ? 0.0 : A.val[k];
    }
    if (!ins) {
      B.col[p] = i;
      B.val[p++] = dmax > 0 ? dmax : 1.0;
    }
  }
  A = std::move(B);
}

}  // namespace

void AMG::setup(const BlockCSR& Ab, const std::vector<double>& mask,
                const std::vector<std::array<double, 3>>* coords) {
  levels_.clear();
  const int bs0 = Ab.B, nn = Ab.n;
  const bool hasmask = !mask.empty();
  // ---- fine level (masked operator as an explicit scalar CSR)
  Level L0;
  L0.bs = bs0;
  CSR& A = L0.A;
  A.nrow = A.ncol = nn * bs0;
  A.ptr.assign(A.nrow + 1, 0);
  const auto& P = *Ab.ptr;
  const auto& C = *Ab.col;
  for (int i = 0; i < nn; ++i)
    for (int a = 0; a < bs0; ++a) {
      int r = i * bs0 + a;
      bool fixed = hasmask && mask[r] == 0.0;
      A.ptr[r + 1] = fixed ? 1 : (P[i + 1] - P[i]) * bs0;
    }
  for (int r = 0; r < A.nrow; ++r) A.ptr[r + 1] += A.ptr[r];
  A.col.resize(A.ptr.back());
  A.val.resize(A.ptr.back());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nn; ++i)
    for (int a = 0; a < bs0; ++a) {
      int r = i * bs0 + a, p = A.ptr[r];
      if (hasmask && mask[r] == 0.0) {
        A.col[p] = r;
        A.val[p] = 1.0;
        continue;
      }
      for (int k = P[i]; k < P[i + 1]; ++k)
        for (int b = 0; b < bs0; ++b) {
          int c = C[k] * bs0 + b;
          double v = Ab.val[size_t(k) * bs0 * bs0 + a * bs0 + b];
          if (hasmask && mask[c] == 0.0) v = 0.0;
          A.col[p] = c;
          A.val[p++] = v;
        }
    }
  // near-nullspace
  int k = (bs0 == 3) ? 6 : 1;
  std::vector<double> Bns(size_t(A.nrow) * k, 0.0);
  if (bs0 == 3) {
    if (!coords) throw std::runtime_error("AMG: elasticity needs node coordinates");
    double c0[3] = {0, 0, 0}, L = 0;
    for (auto& x : *coords)
      for (int d = 0; d < 3; ++d) c0[d] += x[d] / coords->size();
    for (auto& x : *coords)
      for (int d = 0; d < 3; ++d) L = std::max(L, std::abs(x[d] - c0[d]));
    if (L == 0) L = 1;
    for (int i = 0; i < nn; ++i) {
      double x = ((*coords)[i][0] - c0[0]) / L, y = ((*coords)[i][1] - c0[1]) / L, z = ((*coords)[i][2] - c0[2]) / L;
      double rows[3][6] = {{1, 0, 0, 0, z, -y}, {0, 1, 0, -z, 0, x}, {0, 0, 1, y, -x, 0}};
      for (int a = 0; a < 3; ++a)
        for (int j = 0; j < 6; ++j) Bns[size_t(3 * i + a) * 6 + j] = rows[a][j];
    }
  } else {
    std::fill(Bns.begin(), Bns.end(), 1.0);
  }
  if (hasmask)
    for (int r = 0; r < A.nrow; ++r)
      if (mask[r] == 0.0)
        for (int j = 0; j < k; ++j) Bns[size_t(r) * k + j] = 0.0;
  L0.dinv = inv_diag(L0.A);
  L0.lmax = lambda_max_DinvA(L0.A, L0.dinv);
  levels_.push_back(std::move(L0));

  // ---- coarsening
  for (int lev = 0; lev + 1 < prm.max_levels; ++lev) {
    Level& F = levels_.back();
    const int bs = F.bs, n = F.A.nrow, nnode = n / bs;
    if (n <= prm.max_coarse) break;
    const CSR& Af = F.A;
    // block norms
    std::vector<double> dnorm(nnode, 0.0);
#pragma omp parallel for
    for (int I = 0; I < nnode; ++I) {
      double s = 0;
      for (int a = 0; a < bs; ++a) {
        int r = I * bs + a;
        for (int q = Af.ptr[r]; q < Af.ptr[r + 1]; ++q)
          if (Af.col[q] / bs == I) s += Af.val[q] * Af.val[q];
      }
      dnorm[I] = std::sqrt(s);
    }
    std::vector<std::vector<int>> strong(nnode);
    const double theta = prm.theta * std::pow(0.5, lev);
#pragma omp parallel
    {
      std::vector<double> acc(nnode, 0.0);
      std::vector<int> mark(nnode, -1), list;
#pragma omp for schedule(dynamic, 256)
      for (int I = 0; I < nnode; ++I) {
        list.clear();
        for (int a = 0; a < bs; ++a) {
          int r = I * bs + a;
          for (int q = Af.ptr[r]; q < Af.ptr[r + 1]; ++q) {
            int J = Af.col[q] / bs;
            if (J == I) continue;
            if (mark[J] != I) {
              mark[J] = I;
              acc[J] = 0;
              list.push_back(J);
            }
            acc[J] += Af.val[q] * Af.val[q];
          }
        }
        for (int J : list)
          if (std::sqrt(acc[J]) > theta * std::sqrt(dnorm[I] * dnorm[J])) strong[I].push_back(J);
      }
    }
    // aggregation (three phases)
    std::vector<int> agg(nnode, -1);
    int nagg = 0;
    for (int I = 0; I < nnode; ++I) {
      if (agg[I] != -1 || strong[I].empty()) continue;
      bool free = true;
      for (int J : strong[I])
        if (agg[J] != -1) {
          free = false;
          break;
        }
      if (!free) continue;
      agg[I] = nagg;
      for (int J : strong[I]) agg[J] = nagg;
      ++nagg;
    }
    std::vector<int> agg1 = agg;
    for (int I = 0; I < nnode; ++I) {
      if (agg[I] != -1 || strong[I].empty()) continue;
      for (int J : strong[I])
        if (agg1[J] >= 0) {
          agg[I] = agg1[J];
          break;
        }
    }
    for (int I = 0; I < nnode; ++I) {
      if (agg[I] != -1 || strong[I].empty()) continue;
      agg[I] = nagg;
      for (int J : strong[I])
        if (agg[J] == -1) agg[J] = nagg;
      ++nagg;
    }
    if (nagg == 0 || nagg * k >= n) break;  // no coarsening possible
    // tentative prolongator by local QR of the nullspace
    std::vector<std::vector<int>> members(nagg);
    for (int I = 0; I < nnode; ++I)
      if (agg[I] >= 0) members[agg[I]].push_back(I);
    const int nc = nagg * k;
    std::vector<double> Bc(size_t(nc) * k, 0.0);
    // Ptent rows: each fine dof belongs to at most one aggregate -> k entries
    CSR Pt;
    Pt.nrow = n;
    Pt.ncol = nc;
    Pt.ptr.assign(n + 1, 0);
    for (int I = 0; I < nnode; ++I)
      if (agg[I] >= 0)
        for (int a = 0; a < bs; ++a) Pt.ptr[I * bs + a + 1] = k;
    for (int r = 0; r < n; ++r) Pt.ptr[r + 1] += Pt.ptr[r];
    Pt.col.assign(Pt.ptr.back(), 0);
    Pt.val.assign(Pt.ptr.back(), 0.0);
#pragma omp parallel for schedule(dynamic, 64)
    for (int g = 0; g < nagg; ++g) {
      const auto& mem = members[g];
      const int m = int(mem.size()) * bs;
      std::vector<double> Q(size_t(m) * k);
      for (size_t t = 0; t < mem.size(); ++t)
        for (int a = 0; a < bs; ++a)
          for (int j = 0; j < k; ++j) Q[(t * bs + a) * k + j] = Bns[size_t(mem[t] * bs + a) * k + j];
      std::vector<double> R(size_t(k) * k, 0.0);
      for (int j = 0; j < k; ++j) {
        double n0 = 0;
        for (int i = 0; i < m; ++i) n0 += Q[size_t(i) * k + j] * Q[size_t(i) * k + j];
        n0 = std::sqrt(n0);
        for (int pass = 0; pass < 2; ++pass)
          for (int c = 0; c < j; ++c) {
            double d = 0;
            for (int i = 0; i < m; ++i) d += Q[size_t(i) * k + c] * Q[size_t(i) * k + j];
            for (int i = 0; i < m; ++i) Q[size_t(i) * k + j] -= d * Q[size_t(i) * k + c];
            R[c * k + j] += d;
          }
        double nj = 0;
        for (int i = 0; i < m; ++i) nj += Q[size_t(i) * k + j] * Q[size_t(i) * k + j];
        nj = std::sqrt(nj);
        if (nj <= 1e-10 * std::max(n0, 1e-300) || n0 == 0) {
          for (int i = 0; i < m; ++i) Q[size_t(i) * k + j] = 0.0;
          R[j * k + j] = 0.0;
        } else {
          for (int i = 0; i < m; ++i) Q[size_t(i) * k + j] /= nj;
          R[j * k + j] = nj;
        }
      }
      for (size_t t = 0; t < mem.size(); ++t)
        for (int a = 0; a < bs; ++a) {
          int r = mem[t] * bs + a;
          int p = Pt.ptr[r];
          for (int j = 0; j < k; ++j) {
            Pt.col[p + j] = g * k + j;
            Pt.val[p + j] = Q[(t * bs + a) * k + j];
          }
        }
      for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) Bc[size_t(g * k + i) * k + j] = R[i * k + j];
    }
    // smoothed prolongator P = (I - w D^-1 A) Pt
    const double omega = 4.0 / 3.0 / F.lmax;
    CSR AP = spgemm(Af, Pt);
    CSR Pm = AP;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < n; ++r) {
      int q = Pt.ptr[r];
      for (int p = Pm.ptr[r]; p < Pm.ptr[r + 1]; ++p) {
        double v = -omega * F.dinv[r] * AP.val[p];
        while (q < Pt.ptr[r + 1] && Pt.col[q] < Pm.col[p]) ++q;
        if (q < Pt.ptr[r + 1] && Pt.col[q] == Pm.col[p]) v += Pt.val[q];
        Pm.val[p] = v;
      }
    }
    Level Cl;
    Cl.bs = k;
    CSR Rm = Pm.transpose();
    CSR AP2 = spgemm(Af, Pm);
    Cl.A = spgemm(Rm, AP2);
    fix_diagonal(Cl.A);
    Cl.dinv = inv_diag(Cl.A);
    Cl.lmax = lambda_max_DinvA(Cl.A, Cl.dinv);
    F.P = std::move(Pm);
    F.R = std::move(Rm);
    Bns.swap(Bc);
    levels_.push_back(std::move(Cl));
  }
  // ---- coarsest: dense Cholesky
  const CSR& Ac = levels_.back().A;
  nc_ = Ac.nrow;
  if (nc_ > 8000) throw std::runtime_error("AMG: coarsest level too large (" + std::to_string(nc_) + ")");
  chol_.assign(size_t(nc_) * nc_, 0.0);
  double dmax = 0;
  for (int i = 0; i < nc_; ++i)
    for (int q = Ac.ptr[i]; q < Ac.ptr[i + 1]; ++q) {
      chol_[size_t(i) * nc_ + Ac.col[q]] = Ac.val[q];
      if (Ac.col[q] == i) dmax = std::max(dmax, Ac.val[q]);
    }
  for (int j = 0; j < nc_; ++j) {
    double s = chol_[size_t(j) * nc_ + j];
    for (int p = 0; p < j; ++p) s -= chol_[size_t(j) * nc_ + p] * chol_[size_t(j) * nc_ + p];
    if (s <= 1e-14 * dmax) s = std::max(1e-14 * dmax, 1e-300);  // semi-definite guard
    double d = std::sqrt(s);
    chol_[size_t(j) * nc_ + j] = d;
#pragma omp parallel for schedule(static)
    for (int i = j + 1; i < nc_; ++i) {
      double t = chol_[size_t(i) * nc_ + j];
      for (int p = 0; p < j; ++p) t -= chol_[size_t(i) * nc_ + p] * chol_[size_t(j) * nc_ + p];
      chol_[size_t(i) * nc_ + j] = t / d;
    }
  }
}

double AMG::operator_complexity() const {
  double s = 0;
  for (auto& L : levels_) s += L.A.val.size();
  return levels_.empty() ? 0 : s / levels_[0].A.val.size();
}

void AMG::smooth(const Level& L, const std::vector<double>& b, std::vector<double>& x, bool zero) const {
  const int n = L.A.nrow;
  const double lmax = 1.1 * L.lmax, lmin = L.lmax / prm.cheb_ratio;
  const double theta = 0.5 * (lmax + lmin), delta = 0.5 * (lmax - lmin);
  const double sigma = theta / delta;
  double rho = 1.0 / sigma;
  std::vector<double> r(n), d(n), Ad(n);
  if (zero) {
    r = b;
  } else {
    L.A.matvec(x.data(), r.data());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) r[i] = b[i] - r[i];
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    d[i] = L.dinv[i] * r[i] / theta;
    x[i] += d[i];
  }
  for (int it = 1; it < prm.cheb_degree; ++it) {
    L.A.matvec(d.data(), Ad.data());
    double rn = 1.0 / (2.0 * sigma - rho);
    const double c1 = rn * rho, c2 = 2.0 * rn / delta;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      r[i] -= Ad[i];
      d[i] = c1 * d[i] + c2 * L.dinv[i] * r[i];
      x[i] += d[i];
    }
    rho = rn;
  }
}

void AMG::vcycle(int l, const std::vector<double>& b, std::vector<double>& x) const {
  const Level& L = levels_[l];
  if (l + 1 == int(levels_.size())) {
    // dense Cholesky solve
    const int n = nc_;
    x = b;
    for (int i = 0; i < n; ++i) {
      double s = x[i];
      for (int p = 0; p < i; ++p) s -= chol_[size_t(i) * n + p] * x[p];
      x[i] = s / chol_[size_t(i) * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
      double s = x[i];
      for (int p = i + 1; p < n; ++p) s -= chol_[size_t(p) * n + i] * x[p];
      x[i] = s / chol_[size_t(i) * n + i];
    }
    return;
  }
  const int n = L.A.nrow;
  std::fill(x.begin(), x.end(), 0.0);
  smooth(L, b, x, true);
  std::vector<double> r(n);
  L.A.matvec(x.data(), r.data());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) r[i] = b[i] - r[i];
  const int nc = L.R.nrow;
  std::vector<double> bc(nc), xc(nc, 0.0);
  L.R.matvec(r.data(), bc.data());
  vcycle(l + 1, bc, xc);
  L.P.matvec(xc.data(), r.data());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) x[i] += r[i];
  smooth(L, b, x, false);
}

void AMG::apply(const std::vector<double>& r, std::vector<double>& z) const {
  if (z.size() != r.size()) z.assign(r.size(), 0.0);
  vcycle(0, r, z);
}

}  // namespace pcz
