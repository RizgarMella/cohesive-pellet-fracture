#include "random_field.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include "config.hpp"

namespace pcz {

double normal_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double WeibullSpec::scale() const { return (mean - threshold) / std::tgamma(1.0 + 1.0 / modulus); }

double WeibullSpec::quantile(double U) const {
  if (dist == "constant") return mean;
  U = std::min(std::max(U, 1e-12), 1.0 - 1e-12);
  return threshold + scale() * std::pow(-std::log1p(-U), 1.0 / modulus);
}

void WeibullSpec::read(const Config& c, const std::string& s, double def_mean, double def_m) {
  dist = c.get_string(s + ".distribution", "weibull");
  mean = c.get_double(s + ".mean", def_mean);
  modulus = c.get_double(s + ".weibull_modulus", def_m);
  threshold = c.get_double(s + ".threshold", 0.0);
  if (dist != "weibull" && dist != "constant") throw std::runtime_error(s + ".distribution must be weibull|constant");
  if (dist == "weibull" && (modulus <= 0 || mean <= threshold))
    throw std::runtime_error(s + ": need weibull_modulus > 0 and mean > threshold");
}

void GaussianField::generate(const double lo[3], const double hi[3], double lc, uint64_t seed, int ppl, long maxp) {
  // kernel k(x) ~ exp(-x^2/(2 s^2)) with s = lc/2 gives covariance exp(-x^2/lc^2)
  const double s = 0.5 * lc;
  h_ = lc / ppl;
  const int halfw = int(std::ceil(3.0 * s / h_));
  long tot = 1;
  for (int d = 0; d < 3; ++d) {
    o_[d] = lo[d] - h_;
    n_[d] = int(std::ceil((hi[d] - lo[d]) / h_)) + 3;
    tot *= n_[d] + 2 * halfw;
  }
  if (tot > maxp)
    throw std::runtime_error("random field grid too large (" + std::to_string(tot) +
                             " points): increase weibull correlation_length or points_per_length");
  // white noise on the padded grid
  int m[3] = {n_[0] + 2 * halfw, n_[1] + 2 * halfw, n_[2] + 2 * halfw};
  std::vector<float> w(size_t(m[0]) * m[1] * m[2]);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (auto& v : w) v = float(nd(rng));
  std::vector<double> ker(2 * halfw + 1);
  double ksq = 0;
  for (int i = -halfw; i <= halfw; ++i) {
    ker[i + halfw] = std::exp(-0.5 * (i * h_) * (i * h_) / (s * s));
    ksq += ker[i + halfw] * ker[i + halfw];
  }
  const double norm = 1.0 / std::sqrt(ksq * ksq * ksq);  // unit variance after 3 separable passes
  // pass along x
  auto idx = [&](int i, int j, int k, const int* dims) { return (size_t(k) * dims[1] + j) * dims[0] + i; };
  int d1[3] = {n_[0], m[1], m[2]};
  std::vector<float> a(size_t(d1[0]) * d1[1] * d1[2]);
#pragma omp parallel for collapse(2)
  for (int k = 0; k < m[2]; ++k)
    for (int j = 0; j < m[1]; ++j)
      for (int i = 0; i < n_[0]; ++i) {
        double acc = 0;
        for (int t = 0; t <= 2 * halfw; ++t) acc += ker[t] * w[idx(i + t, j, k, m)];
        a[idx(i, j, k, d1)] = float(acc);
      }
  w.clear();
  w.shrink_to_fit();
  int d2[3] = {n_[0], n_[1], m[2]};
  std::vector<float> b(size_t(d2[0]) * d2[1] * d2[2]);
#pragma omp parallel for collapse(2)
  for (int k = 0; k < m[2]; ++k)
    for (int j = 0; j < n_[1]; ++j)
      for (int i = 0; i < n_[0]; ++i) {
        double acc = 0;
        for (int t = 0; t <= 2 * halfw; ++t) acc += ker[t] * a[idx(i, j + t, k, d1)];
        b[idx(i, j, k, d2)] = float(acc);
      }
  a.clear();
  a.shrink_to_fit();
  v_.assign(size_t(n_[0]) * n_[1] * n_[2], 0.f);
#pragma omp parallel for collapse(2)
  for (int k = 0; k < n_[2]; ++k)
    for (int j = 0; j < n_[1]; ++j)
      for (int i = 0; i < n_[0]; ++i) {
        double acc = 0;
        for (int t = 0; t <= 2 * halfw; ++t) acc += ker[t] * b[idx(i, j, k + t, d2)];
        v_[idx(i, j, k, n_)] = float(acc * norm);
      }
}

double GaussianField::operator()(double x, double y, double z) const {
  double p[3] = {(x - o_[0]) / h_, (y - o_[1]) / h_, (z - o_[2]) / h_};
  int i0[3];
  double f[3];
  for (int d = 0; d < 3; ++d) {
    double c = std::min(std::max(p[d], 0.0), double(n_[d] - 1) - 1e-9);
    i0[d] = int(std::floor(c));
    f[d] = c - i0[d];
  }
  double out = 0;
  for (int dk = 0; dk < 2; ++dk)
    for (int dj = 0; dj < 2; ++dj)
      for (int di = 0; di < 2; ++di) {
        double wgt = (di ? f[0] : 1 - f[0]) * (dj ? f[1] : 1 - f[1]) * (dk ? f[2] : 1 - f[2]);
        size_t id = (size_t(i0[2] + dk) * n_[1] + (i0[1] + dj)) * n_[0] + (i0[0] + di);
        out += wgt * v_[id];
      }
  return out;
}

}  // namespace pcz
