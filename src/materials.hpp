// UO2 fuel and fill-gas properties (Williamson & Knoll, SMiRT-20, 2009, section 4).
#pragma once

namespace pcz {

class Config;

struct UO2 {
  double E = 219e9;          // Young's modulus [Pa] (Olander 1976)
  double nu = 0.345;         // Poisson's ratio
  double alpha = 10.0e-6;    // thermal expansion [1/K]
  double T_ref = 293.15;     // strain-free temperature [K]
  double density = 10970.0 * 0.95;  // 95 % TD [kg/m3]
  double conductivity_scale = 1.0;  // multiplier on k(T, Bu) for sensitivity work
  double k_const = -1;        // >0 : constant conductivity [W/m/K] (verification)
  double cp_const = -1;       // >0 : constant specific heat [J/kg/K]
  bool burnup_degradation = true;  // Lucuta et al. (1996) multipliers
  bool swelling = true;            // MATPRO solid + gaseous fission-product swelling
  bool gaseous_swelling = true;

  void read(const Config& c);
  double lambda() const { return E * nu / ((1 + nu) * (1 - 2 * nu)); }
  double mu() const { return E / (2 * (1 + nu)); }

  // thermal conductivity [W/m/K] at temperature T [K] and burnup (FIMA, fraction)
  double k(double T, double fima) const;
  double cp(double T) const;  // [J/kg/K] (Fink 2000)
  // swelling: volumetric strain increments
  double solid_swelling(double fima) const;                            // dV/V total
  double gas_swelling_rate(double T, double fima, double dfima) const;  // d(dV/V)
};

// constants for burnup bookkeeping
constexpr double ENERGY_PER_FISSION = 3.204e-11;  // J (200 MeV)
constexpr double U_MASS_FRACTION = 0.8815;        // kg U / kg UO2
constexpr double AVOGADRO = 6.02214076e23;
constexpr double STEFAN_BOLTZMANN = 5.670374e-8;

double helium_conductivity(double T);  // k_g = 0.00264 T^0.71 (Allison et al. 1993)

}  // namespace pcz
