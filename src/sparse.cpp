#include "sparse.hpp"

#include "amg.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pcz {

void BlockCSR::init(int B_, const std::vector<int>& p, const std::vector<int>& c) {
  B = B_;
  ptr = &p;
  col = &c;
  n = int(p.size()) - 1;
  val.assign(size_t(c.size()) * B * B, 0.0);
  diag.assign(n, -1);
  for (int i = 0; i < n; ++i) diag[i] = find(i, i);
}

void BlockCSR::zero() {
  std::fill(val.begin(), val.end(), 0.0);
}

int BlockCSR::find(int row, int c) const {
  auto b = col->begin() + (*ptr)[row], e = col->begin() + (*ptr)[row + 1];
  auto it = std::lower_bound(b, e, c);
  if (it == e || *it != c) return -1;
  return int(it - col->begin());
}

void BlockCSR::matvec(const std::vector<double>& x, std::vector<double>& y) const {
  const int* P = ptr->data();
  const int* C = col->data();
  const double* V = val.data();
  if (B == 1) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      double s = 0;
      for (int k = P[i]; k < P[i + 1]; ++k) s += V[k] * x[C[k]];
      y[i] = s;
    }
  } else if (B == 3) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      double s0 = 0, s1 = 0, s2 = 0;
      for (int k = P[i]; k < P[i + 1]; ++k) {
        const double* b = V + size_t(k) * 9;
        const double* xj = &x[size_t(C[k]) * 3];
        s0 += b[0] * xj[0] + b[1] * xj[1] + b[2] * xj[2];
        s1 += b[3] * xj[0] + b[4] * xj[1] + b[5] * xj[2];
        s2 += b[6] * xj[0] + b[7] * xj[1] + b[8] * xj[2];
      }
      y[3 * i] = s0;
      y[3 * i + 1] = s1;
      y[3 * i + 2] = s2;
    }
  } else {
    throw std::runtime_error("BlockCSR::matvec: unsupported block size");
  }
}

void BlockCSR::add_diagonal(double s) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i)
    for (int a = 0; a < B; ++a) val[size_t(diag[i]) * B * B + a * B + a] += s;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0;
  const long n = long(a.size());
#pragma omp parallel for reduction(+ : s) schedule(static)
  for (long i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

double norm2(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

double norm_inf(const std::vector<double>& a) {
  double m = 0;
  const long n = long(a.size());
#pragma omp parallel for reduction(max : m) schedule(static)
  for (long i = 0; i < n; ++i) m = std::max(m, std::abs(a[i]));
  return m;
}

static bool invert3(const double* a, double* inv) {
  double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
  if (std::abs(det) < 1e-300) return false;
  double id = 1.0 / det;
  inv[0] = (a[4] * a[8] - a[5] * a[7]) * id;
  inv[1] = (a[2] * a[7] - a[1] * a[8]) * id;
  inv[2] = (a[1] * a[5] - a[2] * a[4]) * id;
  inv[3] = (a[5] * a[6] - a[3] * a[8]) * id;
  inv[4] = (a[0] * a[8] - a[2] * a[6]) * id;
  inv[5] = (a[2] * a[3] - a[0] * a[5]) * id;
  inv[6] = (a[3] * a[7] - a[4] * a[6]) * id;
  inv[7] = (a[1] * a[6] - a[0] * a[7]) * id;
  inv[8] = (a[0] * a[4] - a[1] * a[3]) * id;
  return true;
}

PCGResult pcg(const BlockCSR& A, const std::vector<double>& b, std::vector<double>& x, const std::vector<double>& mask,
              double rtol, int maxit, const AMG* M) {
  const int B = A.B, n = A.n;
  const long N = long(n) * B;
  const bool hasmask = !mask.empty();
  // block-Jacobi preconditioner of the masked operator
  std::vector<double> Pinv(size_t(n) * B * B, 0.0);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    const double* d = &A.val[size_t(A.diag[i]) * B * B];
    double* p = &Pinv[size_t(i) * B * B];
    if (B == 1) {
      double v = (hasmask && mask[i] == 0.0) ? 1.0 : d[0];
      p[0] = std::abs(v) > 0 ? 1.0 / v : 1.0;
    } else {
      double m[9];
      for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 3; ++c) {
          double ma = hasmask ? mask[3 * i + a] : 1.0, mc = hasmask ? mask[3 * i + c] : 1.0;
          m[a * 3 + c] = d[a * 3 + c] * ma * mc + ((a == c && ma == 0.0) ? 1.0 : 0.0);
        }
      if (!invert3(m, p)) {
        for (int a = 0; a < 9; ++a) p[a] = 0;
        for (int a = 0; a < 3; ++a) p[a * 4] = m[a * 4] != 0 ? 1.0 / m[a * 4] : 1.0;
      }
    }
  }
  auto applyA = [&](const std::vector<double>& v, std::vector<double>& out, std::vector<double>& tmp) {
    if (!hasmask) {
      A.matvec(v, out);
      return;
    }
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i) tmp[i] = v[i] * mask[i];
    A.matvec(tmp, out);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i) out[i] = out[i] * mask[i] + (1.0 - mask[i]) * v[i];
  };
  auto applyP = [&](const std::vector<double>& r, std::vector<double>& z) {
    if (M) {
      M->apply(r, z);
      return;
    }
    if (B == 1) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i) z[i] = Pinv[i] * r[i];
    } else {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i) {
        const double* p = &Pinv[size_t(i) * 9];
        const double* ri = &r[size_t(i) * 3];
        z[3 * i] = p[0] * ri[0] + p[1] * ri[1] + p[2] * ri[2];
        z[3 * i + 1] = p[3] * ri[0] + p[4] * ri[1] + p[5] * ri[2];
        z[3 * i + 2] = p[6] * ri[0] + p[7] * ri[1] + p[8] * ri[2];
      }
    }
  };

  std::vector<double> r(N), z(N), p(N), q(N), tmp(hasmask ? N : 0), bb(N);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < N; ++i) bb[i] = hasmask ? b[i] * mask[i] : b[i];
  // fixed dofs of x carry the value (I - M) b = 0 for increments
  if (hasmask) {
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i)
      if (mask[i] == 0.0) x[i] = 0.0;
  }
  applyA(x, q, tmp);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < N; ++i) r[i] = bb[i] - q[i];
  double bnorm = norm2(bb);
  PCGResult res;
  if (bnorm == 0) {
    std::fill(x.begin(), x.end(), 0.0);
    res.converged = true;
    return res;
  }
  applyP(r, z);
  p = z;
  double rz = dot(r, z);
  for (int it = 0; it < maxit; ++it) {
    double rn = norm2(r);
    res.rel_res = rn / bnorm;
    res.iters = it;
    if (res.rel_res <= rtol) {
      res.converged = true;
      return res;
    }
    applyA(p, q, tmp);
    double pq = dot(p, q);
    if (pq <= 0) break;  // loss of positive definiteness
    double alpha = rz / pq;
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * q[i];
    }
    applyP(r, z);
    double rz2 = dot(r, z);
    double beta = rz2 / rz;
    rz = rz2;
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i) p[i] = z[i] + beta * p[i];
  }
  res.rel_res = norm2(r) / bnorm;
  res.converged = res.rel_res <= rtol;
  return res;
}

}  // namespace pcz
