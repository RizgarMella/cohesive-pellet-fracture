#include "materials.hpp"

#include <algorithm>
#include <cmath>

#include "config.hpp"

namespace pcz {

void UO2::read(const Config& c) {
  E = c.get_double("material.youngs_modulus", E);
  nu = c.get_double("material.poisson_ratio", nu);
  alpha = c.get_double("material.thermal_expansion", alpha);
  T_ref = c.get_double("material.reference_temperature", T_ref);
  density = c.get_double("material.density", density);
  conductivity_scale = c.get_double("material.conductivity_scale", conductivity_scale);
  k_const = c.get_double("material.constant_conductivity", k_const);
  cp_const = c.get_double("material.constant_specific_heat", cp_const);
  burnup_degradation = c.get_bool("material.burnup_degradation", burnup_degradation);
  swelling = c.get_bool("material.swelling", swelling);
  gaseous_swelling = c.get_bool("material.gaseous_swelling", gaseous_swelling);
}

// Fink (2000), 95 % TD UO2, multiplied by Lucuta, Matzke & Hastings (1996)
// factors for dissolved (FD) and precipitated (FP) fission products and
// radiation damage (FR). Porosity is already contained in the 95 % TD fit.
double UO2::k(double T, double fima) const {
  if (k_const > 0) return k_const * conductivity_scale;
  T = std::min(std::max(T, 298.0), 3120.0);
  double t = T / 1000.0;
  double k95 = 100.0 / (7.5408 + 17.692 * t + 3.6142 * t * t) + 6400.0 / std::pow(t, 2.5) * std::exp(-16.35 / t);
  if (burnup_degradation && fima > 0) {
    double beta = std::max(fima * 100.0, 1e-6);  // at.%
    double w = 1.09 / std::pow(beta, 3.265) + 0.0643 * std::sqrt(T / beta);
    double FD = w * std::atan(1.0 / w);
    double FP = 1.0 + 0.019 * beta / (3.0 - 0.019 * beta) / (1.0 + std::exp(-(T - 1200.0) / 100.0));
    // radiation-damage factor; ramped in over the first 0.1 at.% so that the
    // conductivity is continuous at the start of irradiation
    double FR0 = 1.0 - 0.2 / (1.0 + std::exp((T - 900.0) / 80.0));
    double ramp = std::min(1.0, beta / 0.1);
    double FR = 1.0 - ramp * (1.0 - FR0);
    k95 *= FD * FP * FR;
  }
  return k95 * conductivity_scale;
}

double UO2::cp(double T) const {
  if (cp_const > 0) return cp_const;
  const double C1 = 302.27, C2 = 8.463e-3, C3 = 8.741e7, th = 548.68, Ea = 18531.7;
  T = std::max(T, 250.0);
  double e = std::exp(th / T);
  return C1 * th * th * e / (T * T * (e - 1) * (e - 1)) + 2 * C2 * T + C3 * Ea / (T * T) * std::exp(-Ea / T);
}

// MATPRO (Allison et al. 1993) forms, burnup in FIMA, density in kg/m3.
double UO2::solid_swelling(double fima) const {
  if (!swelling) return 0.0;
  return 5.577e-5 * density * fima;
}

double UO2::gas_swelling_rate(double T, double fima, double dfima) const {
  if (!swelling || !gaseous_swelling || dfima <= 0) return 0.0;
  double d = std::max(2800.0 - T, 0.0);
  return 1.96e-31 * density * dfima * std::pow(d, 11.73) * std::exp(-0.0162 * d) * std::exp(-0.021 * density * fima);
}

double helium_conductivity(double T) { return 0.00264 * std::pow(std::max(T, 1.0), 0.71); }

}  // namespace pcz
