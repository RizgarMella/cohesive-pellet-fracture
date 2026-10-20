// Constitutive law of the zero-thickness interfaces.
//
// Cohesive interfaces (potential crack planes)
//   * penalty elastic response with stiffness K (Abaqus "K_n") until the
//     effective separation reaches delta_0 = sigma_t / K;
//   * linear (triangular, as in the paper) or exponential softening, with the
//     area under the curve equal to G_c, so delta_f = 2 G_c / sigma_t (linear);
//   * effective separation  delta_e = sqrt(<dn>^2 + beta^2 |ds|^2), beta = 0 is
//     pure mode I (the paper's assumption);
//   * irreversible damage driven by max delta_e; viscous regularisation
//     (Abaqus form)  d_v' = (d - d_v)/mu  integrated with backward Euler;
//   * compression always carried by the full penalty (no interpenetration);
//   * once damaged, closed faces carry Coulomb friction (penalty/return-map),
//     blended as (1-d) cohesive shear + d friction.
// Contact interfaces (pellet-pellet end faces)
//   * unilateral penalty contact with an optional initial gap (dish), Coulomb
//     friction (mu < 0 means full stick).
// The matrix used by the solver is a symmetric secant/stick approximation.
#pragma once
#include <array>

namespace pcz {

struct CohesiveParams {
  double K = 1.0e16;        // penalty / initial stiffness [N/m3] (paper: 1e16)
  double Ks_ratio = 1.0;    // shear stiffness / K
  double beta = 0.0;        // shear weight in the effective separation
  int softening = 0;        // 0 linear, 1 exponential
  double viscosity = 1e-4;  // mu [s] (viscous regularisation, Abaqus convention)
  double friction = 0.0;    // Coulomb coefficient on cracked closed faces
  double residual = 1e-6;   // matrix-only stiffness fraction kept on failed / open faces
  double d_crack = 0.99;    // damage at which a point counts as cracked
};

struct ContactParams {
  double K = 1.0e16;         // normal penalty [N/m3]
  double Kt_ratio = 1.0;     // tangential penalty / K
  double friction = 0.5;     // Coulomb coefficient (<0 : stick)
  double residual = 1e-6;    // matrix-only stiffness fraction when open
};

// properties of one integration point
struct GPProps {
  double sigma_t = 130e6;  // strength [Pa]
  double Gc = 4.0;         // fracture energy [J/m2]
  double gap0 = 0.0;       // initial normal gap [m] (contact interfaces)
};

// history of one integration point (committed values)
struct GPState {
  double dmax = 0.0;   // max effective separation reached [m]
  double d = 0.0;      // rate-independent damage (from dmax)
  double dv = 0.0;     // viscous-regularised damage (used for tractions)
  double s1 = 0.0, s2 = 0.0;  // plastic (slip) tangential separation [m]
};

struct LawOut {
  double t[3];        // local traction (n, t1, t2) [Pa]
  double D[3][3];     // matrix stiffness (local) [Pa/m]
  bool open = false;
  bool slipping = false;
  bool contact = false;
};

// Cohesive law. `st` holds the committed state on input; the trial state is
// written to `trial`. dt = time increment [s].
void cohesive_law(const CohesiveParams& P, const GPProps& g, const GPState& st, const double dl[3], double dt,
                  GPState& trial, LawOut& out);
void contact_law(const ContactParams& P, const GPProps& g, const GPState& st, const double dl[3], GPState& trial,
                 LawOut& out);

// derived softening quantities
double delta0(const CohesiveParams& P, const GPProps& g);
double deltaf(const CohesiveParams& P, const GPProps& g);  // linear law; for exp: characteristic decay length
double damage_from_separation(const CohesiveParams& P, const GPProps& g, double de);
double dissipated_energy(const CohesiveParams& P, const GPProps& g, double dmax);  // [J/m2]
// smallest G_c that still gives a softening branch for this strength
double min_fracture_energy(const CohesiveParams& P, double sigma_t);

}  // namespace pcz
