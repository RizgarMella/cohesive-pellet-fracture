// Weibull-distributed fracture properties.
//
// Two ways of attaching scatter to the cohesive integration points:
//  * correlated (correlation_length > 0): a standard Gaussian random field
//    xi(x) with squared-exponential covariance exp(-|dx|^2 / lc^2) is built on a
//    background grid that does NOT depend on the FE mesh, then mapped point-wise
//    to the Weibull distribution through U = Phi(xi). Refining the mesh samples
//    the same material, so crack-area convergence studies are meaningful
//    ("grain field" of fixed physical size).
//  * uncorrelated (correlation_length = 0): each interface element draws an
//    independent value; optionally with weakest-link size scaling
//    sigma_0,el = sigma_0 (A_ref / A_el)^(1/m) so that the strength of a
//    surface of given area is mesh-independent in the Weibull sense.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace pcz {

class Config;

struct WeibullSpec {
  std::string dist = "weibull";  // weibull | constant
  double mean = 0;               // mean of the distribution (sets the scale)
  double modulus = 8;            // Weibull modulus m
  double threshold = 0;          // location parameter (3-parameter Weibull)
  double scale() const;          // Weibull scale from mean, modulus, threshold
  double quantile(double U) const;
  void read(const Config& c, const std::string& section, double def_mean, double def_m);
};

class GaussianField {
public:
  // grid covering [lo, hi] (padded), spacing <= lc/points_per_lc
  void generate(const double lo[3], const double hi[3], double lc, uint64_t seed, int points_per_lc = 4,
                long max_points = 400000000L);
  double operator()(double x, double y, double z) const;  // trilinear interpolation
  bool empty() const { return v_.empty(); }
  long npoints() const { return long(v_.size()); }

private:
  double o_[3] = {0, 0, 0}, h_ = 1;
  int n_[3] = {0, 0, 0};
  std::vector<float> v_;
};

double normal_cdf(double x);

}  // namespace pcz
