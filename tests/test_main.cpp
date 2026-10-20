// Unit and verification tests (ctest). Each check prints PASS/FAIL.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "config.hpp"
#include "fe.hpp"
#include "interface_law.hpp"
#include "model.hpp"
#include "random_field.hpp"

using namespace pcz;

static int failures = 0;
static void check(bool ok, const std::string& what, double got, double want) {
  std::printf("%s  %-62s got %-12.6g expected %-12.6g\n", ok ? "PASS" : "FAIL", what.c_str(), got, want);
  if (!ok) ++failures;
}
static bool rel(double a, double b, double tol) { return std::abs(a - b) <= tol * std::abs(b); }

// Paper section 3: sigma_t = 130 MPa, Gc = 4 J/m2, K = 1e16 N/m3 -> delta_f = 0.0615 um
static void test_cohesive_law() {
  CohesiveParams P;
  P.viscosity = 0;
  GPProps g;
  check(rel(deltaf(P, g), 0.0615e-6, 0.002), "triangular law: delta_f = 2 Gc / sigma_t (paper: 0.0615 um)", deltaf(P, g),
        0.0615e-6);
  GPState s, tr;
  LawOut o;
  double peak = 0, work = 0, dprev = 0, tprev = 0;
  for (int i = 1; i <= 4000; ++i) {
    double d = 0.07e-6 * i / 4000.0;
    double dl[3] = {d, 0, 0};
    cohesive_law(P, g, s, dl, 1.0, tr, o);
    s = tr;
    peak = std::max(peak, o.t[0]);
    work += 0.5 * (o.t[0] + tprev) * (d - dprev);
    dprev = d;
    tprev = o.t[0];
  }
  check(rel(peak, 130e6, 0.002), "peak traction = sigma_t", peak, 130e6);
  check(rel(work, 4.0, 0.005), "work of separation = Gc", work, 4.0);
  check(s.d == 1.0 && tprev == 0.0, "fully damaged at delta > delta_f", s.d, 1.0);
  check(rel(dissipated_energy(P, g, 1e-6), 4.0, 1e-9), "dissipated energy after failure = Gc", dissipated_energy(P, g, 1e-6), 4.0);
  // unloading is secant and compression uses the full penalty
  GPState half;
  double dl1[3] = {0.04e-6, 0, 0};
  cohesive_law(P, g, GPState{}, dl1, 1.0, half, o);
  double dl2[3] = {0.02e-6, 0, 0};
  cohesive_law(P, g, half, dl2, 1.0, tr, o);
  double secant = (1 - half.d) * P.K * 0.02e-6;
  check(rel(o.t[0], secant, 1e-9), "secant unloading", o.t[0], secant);
  double dl3[3] = {-0.01e-6, 0, 0};
  cohesive_law(P, g, half, dl3, 1.0, tr, o);
  check(rel(o.t[0], -P.K * 0.01e-6, 1e-9), "compression: full penalty after damage", o.t[0], -P.K * 0.01e-6);
  // viscous regularisation lags the damage
  CohesiveParams Pv = P;
  Pv.viscosity = 1e-4;
  double dl4[3] = {0.05e-6, 0, 0};
  cohesive_law(Pv, g, GPState{}, dl4, 1e-4, tr, o);
  check(rel(tr.dv, 0.5 * tr.d, 1e-9), "viscous damage: dv = d dt/mu / (1 + dt/mu)", tr.dv, 0.5 * tr.d);
}

static void test_friction() {
  ContactParams C;
  C.friction = 0.5;
  GPProps g;
  GPState s, tr;
  LawOut o;
  double dl[3] = {-1e-9, 1e-6, 0};  // pressure 1e7 Pa, large slip
  contact_law(C, g, s, dl, tr, o);
  check(rel(o.t[1], 0.5 * 1e7, 1e-9) && o.slipping, "Coulomb: |t_s| = mu p while sliding", o.t[1], 0.5e7);
  double dl2[3] = {1e-9, 0, 0};
  contact_law(C, g, s, dl2, tr, o);
  check(o.t[0] == 0 && o.open, "contact opens without tension", o.t[0], 0);
  g.gap0 = 2e-9;
  contact_law(C, g, s, dl, tr, o);
  check(o.t[0] == 0, "initial gap (dish) prevents contact", o.t[0], 0);
}

static void test_weibull() {
  WeibullSpec w;
  w.mean = 130e6;
  w.modulus = 6;
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> U(0, 1);
  double s = 0, s2 = 0;
  const int n = 400000;
  for (int i = 0; i < n; ++i) {
    double v = w.quantile(U(rng));
    s += v;
    s2 += v * v;
  }
  double mean = s / n, cv = std::sqrt(s2 / n - mean * mean) / mean;
  double cv_exact = std::sqrt(std::tgamma(1 + 2.0 / 6) / std::pow(std::tgamma(1 + 1.0 / 6), 2) - 1);
  check(rel(mean, 130e6, 0.003), "Weibull sample mean = specified mean", mean, 130e6);
  check(rel(cv, cv_exact, 0.01), "Weibull CV(m=6)", cv, cv_exact);
}

static void test_random_field() {
  GaussianField f;
  double lo[3] = {0, 0, 0}, hi[3] = {10e-3, 10e-3, 10e-3};
  double lc = 0.5e-3;
  f.generate(lo, hi, lc, 42, 4);
  double s = 0, s2 = 0, c = 0;
  int n = 0;
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> U(1e-3, 8e-3);
  for (int i = 0; i < 200000; ++i) {
    double x = U(rng), y = U(rng), z = U(rng);
    double a = f(x, y, z), b = f(x + lc, y, z);
    s += a;
    s2 += a * a;
    c += a * b;
    ++n;
  }
  double var = s2 / n - (s / n) * (s / n);
  double corr = (c / n - (s / n) * (s / n)) / var;
  check(std::abs(var - 1.0) < 0.1, "Gaussian field variance ~ 1", var, 1.0);
  check(std::abs(corr - std::exp(-1.0)) < 0.08, "correlation at lag lc ~ exp(-1)", corr, std::exp(-1.0));
}

static std::string base_input(const std::string& extra) {
  return std::string(R"(
[geometry]
radius = 4.67e-3
height = 14e-3
[mesh]
nr = 10
ntheta = 32
nz = 16
[cracks]
radial_planes = 0
[strength]
distribution = constant
[toughness]
distribution = constant
[output]
vtu = false
)") + extra;
}

// Thermo-elastic verification: steady parabolic temperature, generalised plane
// strain hoop stress  sigma_tt = alpha E dT /(4(1-nu)) (3 rho^2 - 1).
static void test_thermoelastic() {
  Config c;
  c.parse_text(base_input(R"(
[material]
constant_conductivity = 3.0
swelling = false
[thermal]
fixed_surface_temperature = true
coolant_temperature = 0, 600
[loading]
power_history_W_per_cm = 0, 200
)"));
  Model m(c);
  m.setup();
  int it = 0;
  bool ok = m.solve_thermal(1.0, 0.0, it);
  double q = 200 * 100.0;
  double dT_exact = q / (4 * M_PI * 3.0);
  double dT = m.T[m.centre_node] - 600.0;
  // the polygonal cross-section has a slightly smaller area -> slightly lower dT
  check(ok && rel(dT, dT_exact, 0.02), "steady centre temperature rise q'/(4 pi k)", dT, dT_exact);
  Diagnostics dg;
  m.fima_trial = 0;
  bool okm = m.solve_mechanics(1.0, 0.0, dg);
  double E = m.mat.E, nu = m.mat.nu, al = m.mat.alpha;
  double err2 = 0, ref2 = 0;
  for (size_t e = 0; e < m.mesh.nelem(); ++e) {
    double cz = 0, cx = 0, cy = 0;
    for (int a = 0; a < 8; ++a) {
      cx += m.mesh.X[m.mesh.hex[e][a]][0] / 8;
      cy += m.mesh.X[m.mesh.hex[e][a]][1] / 8;
      cz += m.mesh.X[m.mesh.hex[e][a]][2] / 8;
    }
    if (std::abs(cz - 7e-3) > 0.5e-3) continue;
    double sg[6];
    m.element_stress(int(e), sg);
    double th = std::atan2(cy, cx), r = std::hypot(cx, cy) / 4.67e-3;
    double hoop = sg[0] * std::sin(th) * std::sin(th) + sg[1] * std::cos(th) * std::cos(th) -
                  2 * sg[3] * std::sin(th) * std::cos(th);
    double ex = al * E * dT_exact / (4 * (1 - nu)) * (3 * r * r - 1);
    err2 += (hoop - ex) * (hoop - ex);
    ref2 += ex * ex;
  }
  double rms = std::sqrt(err2 / ref2);
  check(okm && rms < 0.08, "mid-plane hoop stress vs analytic (relative RMS)", rms, 0.0);
}

// Uniaxial tension across one transverse cohesive plane (Fig. 2 test at
// structural scale): peak force = sigma_t A, dissipated energy = Gc A.
static void test_tension_bar() {
  namespace fs = std::filesystem;
  std::string dir = (fs::temp_directory_path() / "pelletcz_test_bar").string();
  Config c;
  c.parse_text(base_input(R"(
[geometry]
height = 1e-3
[mesh]
nr = 4
ntheta = 16
nz = 2
[cracks]
transverse_z = 0.5
[material]
reference_temperature = 500
[thermal]
coolant_temperature = 0, 500
[boundary]
bottom = roller
top = displacement
top_displacement = 0, 0; 1, 1.0e-6
[time]
end = 1
dt_initial = 0.01
dt_max = 0.01
[cohesive]
viscosity = 1e-5
[output]
directory = )") + dir + "\n");
  Model m(c);
  m.run();
  // read history
  std::ifstream f(dir + "/history.csv");
  std::string header, line;
  std::getline(f, header);
  std::vector<std::string> cols;
  {
    std::stringstream ss(header);
    std::string t;
    while (std::getline(ss, t, ',')) cols.push_back(t);
  }
  int ir = -1, ie = -1;
  for (size_t i = 0; i < cols.size(); ++i) {
    if (cols[i] == "top_reaction_N") ir = int(i);
    if (cols[i] == "dissipated_energy_mJ") ie = int(i);
  }
  double peak = 0, diss = 0, final_force = 0;
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    std::string t;
    std::vector<double> v;
    while (std::getline(ss, t, ',')) v.push_back(std::atof(t.c_str()));
    peak = std::max(peak, v[ir]);
    diss = v[ie] * 1e-3;
    final_force = v[ir];
  }
  const int n = 16;
  double A = 0.5 * n * 4.67e-3 * 4.67e-3 * std::sin(2 * M_PI / n);  // polygon area
  check(rel(peak, 130e6 * A, 0.03), "tension bar: peak force = sigma_t A", peak, 130e6 * A);
  check(rel(diss, 4.0 * A, 0.03), "tension bar: dissipated energy = Gc A", diss, 4.0 * A);
  check(std::abs(final_force) < 1e-3 * peak, "tension bar: no force after separation", final_force, 0.0);
  fs::remove_all(dir);
}

// Two pressed pellets, isothermal: the pellet-pellet contact force must
// equal the applied top pressure times the end-face area, the pins must
// carry nothing, and friction must not slip.
static void test_pressed_stack() {
  namespace fs = std::filesystem;
  std::string dir = (fs::temp_directory_path() / "pelletcz_test_stack").string();
  Config c;
  c.parse_text(base_input(R"(
[geometry]
n_pellets = 2
[mesh]
nr = 4
ntheta = 16
nz = 4
[cracks]
radial_planes = 8
[material]
reference_temperature = 500
[thermal]
coolant_temperature = 0, 500
[interface]
friction = 0.5
contact_fraction = 0.75
[boundary]
bottom = roller
top = pressure
top_pressure = 0, 0; 1, 10e6
[time]
end = 1
dt_initial = 0.25
dt_max = 0.25
[output]
directory = )") + dir + "\n");
  Model m(c);
  m.run();
  const int n = 16;
  double A = 0.5 * n * 4.67e-3 * 4.67e-3 * std::sin(2 * M_PI / n);
  check(rel(m.met.if_force, 10e6 * A, 1e-3), "pressed stack: contact force = p A", m.met.if_force, 10e6 * A);
  double Aann = A * 0.75;
  check(m.met.if_area < 1.05 * Aann && m.met.if_area > 0.6 * Aann, "pressed stack: contact only on the annulus",
        m.met.if_area, Aann);
  check(m.met.pin_reaction < 1e-5 * 10e6 * A, "pressed stack: no pin reactions", m.met.pin_reaction, 0.0);
  check(rel(m.met.bottom_reaction, 10e6 * A, 1e-3), "pressed stack: bottom reaction = p A", m.met.bottom_reaction,
        10e6 * A);
  fs::remove_all(dir);
}

static void test_mesh() {
  Config c;
  c.parse_text(base_input(R"(
[cracks]
radial_planes = 8
circumferential_radii = 0.5
n_slabs = 4
[geometry]
n_pellets = 2
)"));
  MeshParams mp;
  mp.read(c);
  Mesh m;
  m.build(mp);
  int cnt[4] = {0, 0, 0, 0};
  for (auto& f : m.ifaces) cnt[f.kind]++;
  // radial: 8 planes x nr x nz per pellet x 2 pellets; transverse: 3 per pellet x nr*ntheta
  check(cnt[IF_RADIAL] == 8 * 10 * 16 * 2, "radial interface count", cnt[IF_RADIAL], 8 * 10 * 16 * 2);
  check(cnt[IF_TRANSVERSE] == 3 * 10 * 32 * 2, "transverse interface count", cnt[IF_TRANSVERSE], 3 * 10 * 32 * 2);
  check(cnt[IF_CIRCUMFERENTIAL] == 32 * 16 * 2, "circumferential interface count", cnt[IF_CIRCUMFERENTIAL], 32 * 16 * 2);
  check(cnt[IF_PELLET] == 10 * 32, "pellet-pellet interface count", cnt[IF_PELLET], 10 * 32);
  // every interface has coincident node pairs and a normal from A to B
  double maxgap = 0;
  for (auto& f : m.ifaces)
    for (int a = 0; a < 4; ++a)
      for (int i = 0; i < 3; ++i) maxgap = std::max(maxgap, std::abs(m.X[f.a[a]][i] - m.X[f.b[a]][i]));
  check(maxgap == 0.0, "interface node pairs coincide", maxgap, 0.0);
}

int main() {
  test_cohesive_law();
  test_friction();
  test_weibull();
  test_random_field();
  test_mesh();
  test_thermoelastic();
  test_tension_bar();
  test_pressed_stack();
  std::printf("\n%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
