#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "fe.hpp"

namespace pcz {

double Table::operator()(double x) const {
  if (t.empty()) return 0.0;
  if (x <= t.front()) return v.front();
  if (x >= t.back()) return v.back();
  auto it = std::upper_bound(t.begin(), t.end(), x);
  size_t i = size_t(it - t.begin());
  double f = (x - t[i - 1]) / (t[i] - t[i - 1]);
  return v[i - 1] + f * (v[i] - v[i - 1]);
}

static Table read_table(const Config& c, const std::string& key, double scale, double def) {
  Table T;
  auto g = c.get_groups(key);
  if (g.empty()) {
    T.set_constant(def);
    return T;
  }
  for (auto& row : g) {
    if (row.size() < 2) throw std::runtime_error(key + ": expected 'time, value; time, value; ...'");
    if (!T.t.empty() && row[0] <= T.t.back()) throw std::runtime_error(key + ": times must increase");
    T.t.push_back(row[0]);
    T.v.push_back(row[1] * scale);
  }
  return T;
}

Model::Model(const Config& c) : cfg(c) {
  mat.read(c);
  // cohesive law
  coh.K = c.get_double("cohesive.stiffness", coh.K);
  coh.Ks_ratio = c.get_double("cohesive.shear_stiffness_ratio", coh.Ks_ratio);
  coh.beta = c.get_double("cohesive.mode_mixity_beta", coh.beta);
  std::string soft = c.get_string("cohesive.softening", "linear");
  coh.softening = soft == "exponential" ? 1 : 0;
  if (soft != "linear" && soft != "exponential") throw std::runtime_error("cohesive.softening must be linear|exponential");
  coh.viscosity = c.get_double("cohesive.viscosity", coh.viscosity);
  coh.friction = c.get_double("cohesive.crack_face_friction", coh.friction);
  coh.residual = c.get_double("cohesive.residual_stiffness", coh.residual);
  coh.d_crack = c.get_double("cohesive.cracked_damage", coh.d_crack);
  // fracture statistics
  strength.read(c, "strength", 130e6, 8.0);
  toughness.read(c, "toughness", 4.0, 8.0);
  corr_length = c.get_double("weibull.correlation_length", corr_length);
  tough_corr = c.get_double("weibull.toughness_strength_correlation", tough_corr);
  ref_area = c.get_double("weibull.reference_area", ref_area);
  points_per_length = c.get_int("weibull.points_per_length", points_per_length);
  seed = uint64_t(c.get_int("weibull.seed", 1));
  flaws = c.get_groups("weibull.flaws");
  // pellet interface
  con.K = c.get_double("interface.stiffness", coh.K);
  con.Kt_ratio = c.get_double("interface.tangential_stiffness_ratio", con.Kt_ratio);
  std::string fr = c.get_string("interface.friction", "0.5");
  con.friction = (fr == "stick") ? -1.0 : std::stod(fr);
  con.residual = coh.residual;
  double cf = c.get_double("interface.contact_fraction", 1.0);
  dish_inner_radius = c.get_double("interface.dish_radius", -1.0);
  dish_depth = c.get_double("interface.dish_depth", dish_depth);
  chamfer_width = c.get_double("interface.chamfer_width", chamfer_width);
  chamfer_depth = c.get_double("interface.chamfer_depth", chamfer_depth);
  double R = c.get_double("geometry.radius", 4.67e-3);
  if (dish_inner_radius < 0) dish_inner_radius = (cf < 1.0) ? R * std::sqrt(std::max(0.0, 1.0 - cf)) : 0.0;
  // time
  tp.t_end = c.get_double("time.end", tp.t_end);
  tp.dt0 = c.get_double("time.dt_initial", tp.dt0);
  tp.dt_min = c.get_double("time.dt_min", tp.dt_min);
  tp.dt_max = c.get_double("time.dt_max", tp.dt_max);
  tp.grow = c.get_double("time.grow", tp.grow);
  tp.cut = c.get_double("time.cut", tp.cut);
  tp.max_dpower_frac = c.get_double("time.max_power_change", tp.max_dpower_frac);
  tp.max_damage_increment = c.get_double("time.max_damage_increment", tp.max_damage_increment);
  tp.max_steps = c.get_int("time.max_steps", tp.max_steps);
  tp.softening_dt_factor = c.get_double("time.softening_dt_factor", tp.softening_dt_factor);
  tp.initiation_target = c.get_double("time.initiation_target", tp.initiation_target);
  // solver
  sp.newton_max = c.get_int("solver.newton_max_iterations", sp.newton_max);
  sp.newton_tol = c.get_double("solver.newton_tolerance", sp.newton_tol);
  sp.newton_dtol = c.get_double("solver.displacement_tolerance", sp.newton_dtol);
  sp.lin_rtol = c.get_double("solver.linear_tolerance", sp.lin_rtol);
  sp.lin_maxit = c.get_int("solver.linear_max_iterations", sp.lin_maxit);
  sp.stabilization = c.get_double("solver.stabilization", sp.stabilization);
  sp.thermal_max = c.get_int("solver.thermal_max_iterations", sp.thermal_max);
  sp.thermal_tol = c.get_double("solver.thermal_tolerance", sp.thermal_tol);
  sp.coupling_iters = c.get_int("solver.coupling_iterations", sp.coupling_iters);
  sp.thermal_skip_dt = c.get_double("solver.thermal_skip_dt", sp.thermal_skip_dt);
  sp.line_search = c.get_bool("solver.line_search", sp.line_search);
  sp.amg = c.get_string("solver.preconditioner", "amg") == "amg";
  sp.amg_rebuild_iters = c.get_int("solver.amg_rebuild_iterations", sp.amg_rebuild_iters);
  // thermal BCs
  tbc.h_conv = c.get_double("thermal.convection_coefficient", tbc.h_conv);
  tbc.T_coolant = read_table(c, "thermal.coolant_temperature", 1.0, 500.0);
  tbc.fixed_surface = c.get_bool("thermal.fixed_surface_temperature", tbc.fixed_surface);
  tbc.emissivity = c.get_double("thermal.crack_emissivity", tbc.emissivity);
  tbc.roughness = c.get_double("thermal.crack_roughness", tbc.roughness);
  tbc.h_bonded = c.get_double("thermal.bonded_conductance", tbc.h_bonded);
  tbc.h_contact = c.get_double("thermal.pellet_contact_conductance", tbc.h_contact);
  tbc.T_initial = c.get_double("thermal.initial_temperature", tbc.T_coolant(0.0));
  // mechanical BCs
  mbc.bottom = c.get_string("boundary.bottom", mbc.bottom);
  mbc.platen_friction = c.get_double("boundary.platen_friction", mbc.platen_friction);
  mbc.platen_stiffness = c.get_double("boundary.platen_stiffness", con.K);
  mbc.top = c.get_string("boundary.top", mbc.top);
  mbc.top_pressure = read_table(c, "boundary.top_pressure", 1.0, c.get_double("boundary.pressure", 0.0));
  mbc.top_displacement = read_table(c, "boundary.top_displacement", 1.0, 0.0);
  mbc.pin_rigid_body = c.get_bool("boundary.pin_rigid_body", mbc.pin_rigid_body);
  if (mbc.bottom != "roller" && mbc.bottom != "platen") throw std::runtime_error("boundary.bottom must be roller|platen");
  if (mbc.top != "free" && mbc.top != "pressure" && mbc.top != "displacement")
    throw std::runtime_error("boundary.top must be free|pressure|displacement");
  // power history (W/cm in the input, like the paper)
  power = read_table(c, "loading.power_history_W_per_cm", 100.0, 0.0);
  // outputs
  op.dir = c.get_string("output.directory", op.dir);
  op.times = c.get_list("output.times", {});
  op.every = c.get_int("output.every_steps", op.every);
  op.on_crack_growth = c.get_double("output.on_crack_area_growth", op.on_crack_growth);
  op.vtu = c.get_bool("output.vtu", op.vtu);
  op.binary = c.get_bool("output.binary", op.binary);
  op.max_vtu = c.get_int("output.max_vtu", op.max_vtu);
  op.angular_bins = c.get_int("output.angular_bins", op.angular_bins);
  // time stops: history breakpoints and output times
  tp.stops = op.times;
  for (double t : power.t) tp.stops.push_back(t);
  for (double t : mbc.top_pressure.t) tp.stops.push_back(t);
  for (double t : mbc.top_displacement.t) tp.stops.push_back(t);
  tp.stops.push_back(tp.t_end);
  std::sort(tp.stops.begin(), tp.stops.end());
  tp.stops.erase(std::remove_if(tp.stops.begin(), tp.stops.end(), [&](double t) { return t <= 0 || t > tp.t_end; }),
                 tp.stops.end());
  tp.stops.erase(std::unique(tp.stops.begin(), tp.stops.end()), tp.stops.end());
}

double Model::linear_power(double t) const { return power(t); }

double Model::power_density(double t) const {
  return power(t) / (M_PI * mesh.p.R * mesh.p.R);
}

void Model::setup() {
  MeshParams mp;
  mp.read(cfg);
  mesh.build(mp);
  const int nn = int(mesh.nnode()), ne = int(mesh.nelem());
  std::cout << mesh.summary() << "\n";

  // element volumes
  elem_vol.assign(ne, 0.0);
#pragma omp parallel for
  for (int e = 0; e < ne; ++e) {
    double X[8][3], N[8], B[8][3], dV;
    for (int a = 0; a < 8; ++a)
      for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[mesh.hex[e][a]][i];
    double v = 0;
    for (int g = 0; g < 8; ++g) {
      if (!hex_gp(X, g, N, B, dV)) throw std::runtime_error("non-positive Jacobian in element " + std::to_string(e));
      v += dV;
    }
    elem_vol[e] = v;
  }

  // interface Gauss points
  const size_t ni = mesh.ifaces.size();
  ngp = ni * 4;
  gp_dA.assign(ngp, 0.0);
  gp_R.assign(ngp, {});
  gp_x.assign(ngp, {});
  gp_props.assign(ngp, GPProps{});
  gp_state.assign(ngp, GPState{});
  gp_trial = gp_state;
  gp_open.assign(ngp, 0.f);
  gp_tn.assign(ngp, 0.f);
  gp_slip.assign(ngp, 0.f);
  gp_ts.assign(ngp, 0.f);
  for (size_t f = 0; f < ni; ++f) {
    double X[4][3];
    for (int a = 0; a < 4; ++a)
      for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[mesh.ifaces[f].a[a]][i];
    for (int g = 0; g < 4; ++g) {
      double N[4], dA, R[3][3], x[3];
      quad_gp(X, g, N, dA, R, x);
      size_t q = f * 4 + g;
      gp_dA[q] = dA;
      for (int i = 0; i < 3; ++i) {
        gp_x[q][i] = x[i];
        for (int j = 0; j < 3; ++j) gp_R[q][i * 3 + j] = R[i][j];
      }
    }
  }
  assign_fracture_properties();

  // bottom platen nodes (tributary areas)
  if (mbc.bottom == "platen") {
    std::map<int, double> area;
    for (auto& bf : mesh.bottom) {
      double X[4][3];
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[bf.n[a]][i];
      for (int g = 0; g < 4; ++g) {
        double N[4], dA, R[3][3], x[3];
        quad_gp(X, g, N, dA, R, x);
        for (int a = 0; a < 4; ++a) area[bf.n[a]] += N[a] * dA;
      }
    }
    for (auto& [n, a] : area) {
      platen_node.push_back(n);
      platen_area.push_back(a);
    }
    platen_state.assign(platen_node.size(), GPState{});
    platen_trial = platen_state;
  }

  // Dirichlet masks
  mask_u.assign(size_t(3) * nn, 1.0);
  if (mbc.bottom == "roller")
    for (auto& bf : mesh.bottom)
      for (int a = 0; a < 4; ++a) mask_u[3 * bf.n[a] + 2] = 0.0;
  if (mbc.pin_rigid_body) {
    // statically determinate in-plane pins on the region of element 0:
    // axis node (ux, uy) and the next node along +x (uy): no reactions for
    // self-equilibrated loads.
    int n0 = mesh.hex[0][0], n1 = mesh.hex[0][1];
    pin_dofs = {3 * n0, 3 * n0 + 1, 3 * n1 + 1};
    for (int d : pin_dofs) mask_u[d] = 0.0;
  }
  {
    std::vector<char> isb(nn, 0);
    for (auto& bf : mesh.bottom)
      for (int a = 0; a < 4; ++a) isb[bf.n[a]] = 1;
    for (int n = 0; n < nn; ++n)
      if (isb[n]) bottom_nodes.push_back(n);
    // monitoring points: pellet-0 centre and outer surface at mid-height
    const double zm = 0.5 * mesh.p.H;
    double best = 1e300, zbest = 0;
    for (int n = 0; n < nn; ++n) {
      double d = std::hypot(mesh.X[n][0], mesh.X[n][1]) + std::abs(mesh.X[n][2] - zm);
      if (d < best) {
        best = d;
        centre_node = n;
        zbest = mesh.X[n][2];
      }
    }
    for (int n = 0; n < nn; ++n)
      if (std::abs(mesh.X[n][2] - zbest) < 1e-12 &&
          std::abs(std::hypot(mesh.X[n][0], mesh.X[n][1]) - mesh.p.R) < 1e-9 * mesh.p.R + 1e-12)
        surf_nodes.push_back(n);
  }
  {
    std::vector<char> is_top(nn, 0);
    for (auto& bf : mesh.top)
      for (int a = 0; a < 4; ++a) is_top[bf.n[a]] = 1;
    for (int n = 0; n < nn; ++n)
      if (is_top[n]) top_nodes.push_back(n);
    if (mbc.top == "displacement")
      for (int n : top_nodes) mask_u[3 * n + 2] = 0.0;
  }
  mask_T.clear();
  if (tbc.fixed_surface) {
    mask_T.assign(nn, 1.0);
    for (auto& bf : mesh.outer)
      for (int a = 0; a < 4; ++a) mask_T[bf.n[a]] = 0.0;
  }

  // matrices and scatter maps
  Kb.init(3, mesh.adj_ptr, mesh.adj);
  Kmat.init(3, mesh.adj_ptr, mesh.adj);
  Kth.init(1, mesh.adj_ptr, mesh.adj);
  hex_blk.resize(ne);
#pragma omp parallel for
  for (int e = 0; e < ne; ++e)
    for (int a = 0; a < 8; ++a)
      for (int b = 0; b < 8; ++b) hex_blk[e][a * 8 + b] = Kb.find(mesh.hex[e][a], mesh.hex[e][b]);
  if_blk.resize(ni);
#pragma omp parallel for
  for (long f = 0; f < long(ni); ++f) {
    int ids[8];
    for (int a = 0; a < 4; ++a) {
      ids[a] = mesh.ifaces[f].a[a];
      ids[a + 4] = mesh.ifaces[f].b[a];
    }
    for (int a = 0; a < 8; ++a)
      for (int b = 0; b < 8; ++b) if_blk[f][a * 8 + b] = Kb.find(ids[a], ids[b]);
  }

  // fields
  T.assign(nn, tbc.T_initial);
  T_old = T;
  u.assign(size_t(3) * nn, 0.0);
  u_old = u;
  gas_swell.assign(ne, 0.0);
  gas_swell_trial = gas_swell;
  assemble_bulk_stiffness();
}

void Model::assign_fracture_properties() {
  const size_t ni = mesh.ifaces.size();
  double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
  for (auto& x : mesh.X)
    for (int i = 0; i < 3; ++i) {
      lo[i] = std::min(lo[i], x[i]);
      hi[i] = std::max(hi[i], x[i]);
    }
  GaussianField f1, f2;
  const bool correlated = corr_length > 0;
  const bool need2 = correlated && toughness.dist == "weibull" && tough_corr < 1.0;
  if (correlated) {
    f1.generate(lo, hi, corr_length, seed, points_per_length);
    if (need2) f2.generate(lo, hi, corr_length, seed * 7919 + 17, points_per_length);
    std::cout << "weibull: correlated field, lc = " << corr_length * 1e3 << " mm, " << f1.npoints()
              << " grid points\n";
  }
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  const double rho = std::max(-1.0, std::min(1.0, tough_corr));
  const double rr = std::sqrt(std::max(0.0, 1 - rho * rho));
  n_gc_clamped = 0;
  const double R = mesh.p.R;
  for (size_t f = 0; f < ni; ++f) {
    const auto& ifc = mesh.ifaces[f];
    const bool cohesive = ifc.kind != IF_PELLET || mesh.p.cohesive_pellet_interface;
    double z1 = 0, z2 = 0;
    double Ael = 0;
    if (!correlated) {
      z1 = nd(rng);
      z2 = nd(rng);
      for (int g = 0; g < 4; ++g) Ael += gp_dA[f * 4 + g];
    }
    for (int g = 0; g < 4; ++g) {
      size_t q = f * 4 + g;
      auto& P = gp_props[q];
      const auto& x = gp_x[q];
      double r = std::hypot(x[0], x[1]);
      if (!cohesive) {  // contact interface: geometry only
        P.gap0 = 0.0;
        if (r < dish_inner_radius) P.gap0 = dish_depth;
        if (chamfer_width > 0 && r > R - chamfer_width) P.gap0 = chamfer_depth;
        continue;
      }
      double xi1 = correlated ? f1(x[0], x[1], x[2]) : z1;
      double xi2b = correlated ? (need2 ? f2(x[0], x[1], x[2]) : 0.0) : z2;
      double xiG = rho * xi1 + rr * xi2b;
      double Us = normal_cdf(xi1), Ug = normal_cdf(xiG);
      double sig = strength.quantile(Us);
      if (!correlated && ref_area > 0 && strength.dist == "weibull")
        sig = strength.threshold + (sig - strength.threshold) * std::pow(ref_area / Ael, 1.0 / strength.modulus);
      double gc = toughness.quantile(Ug);
      for (auto& fl : flaws) {  // x, y, z, radius, strength_factor [, toughness_factor]
        if (fl.size() < 5) throw std::runtime_error("weibull.flaws: x,y,z,radius,strength_factor[,toughness_factor]");
        double dx = x[0] - fl[0], dy = x[1] - fl[1], dz = x[2] - fl[2];
        if (dx * dx + dy * dy + dz * dz <= fl[3] * fl[3]) {
          sig *= fl[4];
          if (fl.size() > 5) gc *= fl[5];
        }
      }
      double gmin = 1.05 * min_fracture_energy(coh, sig);
      if (gc < gmin) {
        gc = gmin;
        ++n_gc_clamped;
      }
      P.sigma_t = sig;
      P.Gc = gc;
      P.gap0 = 0.0;
    }
  }
  if (n_gc_clamped)
    std::cout << "warning: G_c raised to 1.05 sigma_t^2/(2K) at " << n_gc_clamped
              << " points (softening branch must exist)\n";
}

}  // namespace pcz
