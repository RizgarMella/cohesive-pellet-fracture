// Time stepping: staggered thermal -> mechanical solves per step with
// adaptive step size. A step is rejected (state rolled back, dt cut) if
// Newton fails or the viscous damage jumps by more than
// time.max_damage_increment anywhere, which resolves the fast crack growth
// events (steps down to microseconds) and lets dt grow to months during
// steady operation - the behaviour shown in Fig. 6 of the paper.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "fe.hpp"
#include "model.hpp"

namespace pcz {

static double wall_now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int Model::label_fragments(std::vector<int>& lab, std::vector<int>& elem_rid) const {
  // region ids
  const int ne = int(mesh.nelem());
  std::map<std::array<int, 4>, int> rid;
  elem_rid.resize(ne);
  for (int e = 0; e < ne; ++e) {
    auto it = rid.find(mesh.elem_region[e]);
    if (it == rid.end()) it = rid.emplace(mesh.elem_region[e], int(rid.size())).first;
    elem_rid[e] = it->second;
  }
  // node -> region (via any element using the node)
  std::vector<int> node_rid(mesh.nnode(), -1);
  for (int e = 0; e < ne; ++e)
    for (int a = 0; a < 8; ++a) node_rid[mesh.hex[e][a]] = elem_rid[e];
  std::vector<int> parent(rid.size());
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };
  for (size_t f = 0; f < mesh.ifaces.size(); ++f) {
    const auto& ifc = mesh.ifaces[f];
    bool cohesive = ifc.kind != IF_PELLET || mesh.p.cohesive_pellet_interface;
    if (!cohesive) continue;
    bool intact = false;
    for (int g = 0; g < 4; ++g)
      if (gp_state[f * 4 + g].d < coh.d_crack) intact = true;
    if (!intact) continue;
    int a = find(node_rid[ifc.a[0]]), b = find(node_rid[ifc.b[0]]);
    if (a != b) parent[a] = b;
  }
  std::map<int, int> roots;
  lab.resize(parent.size());
  for (size_t r = 0; r < parent.size(); ++r) {
    int root = find(int(r));
    auto it = roots.find(root);
    if (it == roots.end()) it = roots.emplace(root, int(roots.size())).first;
    lab[r] = it->second;
  }
  return int(roots.size());
}

void Model::compute_metrics() {
  Metrics m;
  m.pellet_area.assign(mesh.p.n_pellets, 0.0);
  for (size_t f = 0; f < mesh.ifaces.size(); ++f) {
    const auto& ifc = mesh.ifaces[f];
    bool cohesive = ifc.kind != IF_PELLET || mesh.p.cohesive_pellet_interface;
    for (int g = 0; g < 4; ++g) {
      size_t q = f * 4 + g;
      const double dA = gp_dA[q];
      if (cohesive) {
        const auto& s = gp_state[q];
        if (s.d >= coh.d_crack) {
          m.area[ifc.kind] += dA;
          m.pellet_area[ifc.pellet] += dA;
        }
        if (s.d > 0) m.damaged_area += dA;
        m.dissipated += dA * dissipated_energy(coh, gp_props[q], s.dmax);
      } else if (gp_tn[q] < 0) {
        double p = -gp_tn[q];
        m.if_force += p * dA;
        m.if_area += dA;
        m.if_pmax = std::max(m.if_pmax, p);
        m.if_slip_frac += gp_slip[q] * dA;
        m.if_shear_force += gp_ts[q] * dA;
      }
    }
  }
  if (m.if_area > 0) m.if_slip_frac /= m.if_area;
  m.Tc = T[centre_node];
  double s = 0;
  for (int n : surf_nodes) s += T[n];
  m.Ts = surf_nodes.empty() ? T[centre_node] : s / surf_nodes.size();
  m.Tmax = *std::max_element(T.begin(), T.end());
  double mp = -1e300;
  const int ne = int(mesh.nelem());
#pragma omp parallel for reduction(max : mp)
  for (int e = 0; e < ne; ++e) {
    double sig[6], ev[3];
    element_stress(e, sig);
    sym_eig3(sig, ev);
    mp = std::max(mp, ev[0]);
  }
  m.max_principal = mp;
  if (!Rres.empty()) {
    for (int n : bottom_nodes)
      if (mask_u[3 * n + 2] == 0.0) m.bottom_reaction += -Rres[3 * n + 2];
    if (mbc.top == "displacement")
      for (int n : top_nodes) m.top_reaction += -Rres[3 * n + 2];
    double px = 0, py = 0;
    for (int d : pin_dofs) (d % 3 == 0 ? px : py) += -Rres[d];
    m.pin_reaction = std::hypot(px, py);
  }
  if (mbc.bottom == "platen")
    for (size_t i = 0; i < platen_node.size(); ++i) {
      int n = platen_node[i];
      double gap = u[3 * n + 2];
      if (gap < 0) m.bottom_reaction += -mbc.platen_stiffness * gap * platen_area[i];
    }
  std::vector<int> lab, erid;
  m.n_fragments = label_fragments(lab, erid);
  met = m;
}

void Model::run() {
  namespace fs = std::filesystem;
  wall_start = wall_now();
  fs::create_directories(op.dir);
  setup();
  write_manifest();
  open_history();

  // initial state
  Diagnostics dg;
  fima_trial = fima;
  int ith = 0;
  solve_thermal(0.0, 0.0, ith);
  T_old = T;
  solve_mechanics(0.0, 0.0, dg);
  u_old = u;
  gp_state = gp_trial;
  platen_state = platen_trial;
  compute_metrics();
  write_history_row(dg);
  if (op.vtu) write_vtu(0);
  vtu_list.push_back({time, 0});
  write_angular(0);
  write_interface_profile(0);

  double dt = tp.dt0;
  double Pmax = 0;
  for (double v : power.v) Pmax = std::max(Pmax, std::abs(v));
  const double NU = mat.density * U_MASS_FRACTION / 0.238 * AVOGADRO;  // U atoms / m3
  const double eps_t = 1e-9 * std::max(1.0, tp.t_end);
  size_t stop_i = 0;
  int rejections = 0;
  std::string end_msg = "completed";
  bool complete = true;
  double last_print = wall_now();
  double fi_prev = 0.0, t_prev_step = 0.0;

  while (time < tp.t_end - eps_t) {
    if (step >= tp.max_steps) {
      end_msg = "time.max_steps reached";
      complete = false;
      break;
    }
    while (stop_i < tp.stops.size() && tp.stops[stop_i] <= time + eps_t) ++stop_i;
    double t_stop = stop_i < tp.stops.size() ? tp.stops[stop_i] : tp.t_end;
    dt = std::min({dt, tp.dt_max, t_stop - time});
    // limit the power change per step
    if (Pmax > 0)
      for (int k = 0; k < 40; ++k) {
        if (std::abs(power(time + dt) - power(time)) <= tp.max_dpower_frac * Pmax || dt <= tp.dt_min) break;
        dt *= 0.5;
      }
    const double t1 = (t_stop - (time + dt) < 1e-6 * dt) ? t_stop : time + dt;
    dt = t1 - time;

    // ---- trial step
    const double q0 = power_density(time), q1 = power_density(t1);
    const double dfima = 0.5 * (q0 + q1) * dt / (ENERGY_PER_FISSION * NU);
    fima_trial = fima + dfima;
    const int ne = int(mesh.nelem());
#pragma omp parallel for
    for (int e = 0; e < ne; ++e) {
      double Tc = 0;
      for (int a = 0; a < 8; ++a) Tc += T_old[mesh.hex[e][a]];
      gas_swell_trial[e] = gas_swell[e] + mat.gas_swelling_rate(Tc / 8.0, fima, dfima);
    }
    bool ok = true;
    std::string why;
    gp_trial = gp_state;
    for (int pass = 0; pass < std::max(1, sp.coupling_iters) && ok; ++pass) {
      double w0 = wall_now();
      // within a crack burst (dt << thermal time constant, ~R^2 rho c / k ~ 20 s) the
      // temperature field cannot change: keep it and solve only the mechanics
      bool okt = true;
      if (dt < sp.thermal_skip_dt && pass == 0)
        dg.thermal_iters = 0;
      else
        okt = solve_thermal(t1, dt, dg.thermal_iters);
      tm_thermal += wall_now() - w0;
      if (!okt) {
        ok = false;
        why = "thermal";
        break;
      }
      w0 = wall_now();
      bool okm = solve_mechanics(t1, dt, dg);
      tm_mech += wall_now() - w0;
      if (!okm) {
        ok = false;
        why = "newton";
      }
    }
    double maxdd = 0;
    if (ok) {
#pragma omp parallel for reduction(max : maxdd)
      for (long q = 0; q < long(ngp); ++q) maxdd = std::max(maxdd, gp_trial[q].dv - gp_state[q].dv);
      if (maxdd > tp.max_damage_increment && dt > 4 * tp.dt_min) {
        ok = false;
        why = "damage";
      }
    }
    if (!ok) {
      T = T_old;
      u = u_old;
      gp_trial = gp_state;
      platen_trial = platen_state;
      dt *= tp.cut;
      // a failure while cracks soften: go straight to the viscous time scale
      if (n_softening > 0 && coh.viscosity > 0) dt = std::min(dt, 4.0 * tp.softening_dt_factor * coh.viscosity);
      ++rejections;
      if (dt < tp.dt_min) {
        end_msg = "time step below time.dt_min at t = " + std::to_string(time) + " s (" + why + ")";
        complete = false;
        std::cerr << end_msg << std::endl;
        break;
      }
      continue;
    }

    // ---- commit
    T_old = T;
    u_old = u;
    gp_state = gp_trial;
    platen_state = platen_trial;
    gas_swell = gas_swell_trial;
    fima = fima_trial;
    time = t1;
    ++step;
    double w1 = wall_now();
    compute_metrics();
    double area = met.area[0] + met.area[1] + met.area[2];
    if (first_crack_time < 0 && area > 0) {
      first_crack_time = time;
      first_crack_dT = met.Tc - met.Ts;
      first_crack_power = linear_power(time);
      std::cout << "  first crack at t = " << time << " s, dT = " << first_crack_dT << " K, q' = "
                << first_crack_power / 100.0 << " W/cm\n";
    }
    write_history_row(dg);
    tm_metrics += wall_now() - w1;
    bool out = false;
    for (double to : op.times)
      if (std::abs(to - time) <= eps_t) out = true;
    if (op.every > 0 && step % op.every == 0) out = true;
    if (op.on_crack_growth > 0 && area - last_vtu_area >= op.on_crack_growth) out = true;
    if (time >= tp.t_end - eps_t) out = true;
    if (out && n_vtu < op.max_vtu) {
      ++n_vtu;
      if (op.vtu) write_vtu(n_vtu);
      vtu_list.push_back({time, n_vtu});
      write_pvd();  // keep the collections current so partial runs stay usable
      write_angular(n_vtu);
      write_interface_profile(n_vtu);
      last_vtu_area = area;
    }
    if (wall_now() - last_print > 5.0 || out) {
      last_print = wall_now();
      std::cout << std::setprecision(4) << "step " << step << "  t=" << time << " s  dt=" << dt
                << "  q'=" << linear_power(time) / 100.0 << " W/cm  dT=" << met.Tc - met.Ts
                << " K  crack area=" << area * 1e6 << " mm2  frags=" << met.n_fragments
                << "  FI=" << failure_index << " soft=" << n_softening << "  newton=" << dg.newton_iters
                << " pcg=" << dg.lin_iters << std::endl;
    }
    // ---- adapt
    if (dg.newton_iters <= 4 && maxdd < 0.5 * tp.max_damage_increment)
      dt *= tp.grow;
    else if (dg.newton_iters > sp.newton_max / 2)
      dt *= 0.7;
    const double dt_visc = tp.softening_dt_factor * std::max(coh.viscosity, tp.dt_min);
    if (n_softening > 0) {
      dt = std::min(dt, dt_visc);  // cracks are running: resolve them on the viscous time scale
    } else if (failure_index > 0.3 && failure_index > fi_prev && fi_prev > 0) {
      // approach the next initiation: extrapolate the failure index linearly
      double rate = (failure_index - fi_prev) / std::max(time - t_prev_step, 1e-30);
      double dt_pred = (tp.initiation_target - failure_index) / rate;
      // shrink at most 4x per step: weak Weibull points that start to soften
      // stably do not need to be approached on the viscous time scale
      const double dt_last = time - t_prev_step;
      dt = std::min(dt, std::max({dt_pred, 0.25 * dt_last, dt_visc}));
    }
    fi_prev = failure_index;
    t_prev_step = time;
  }
  std::cout << "finished: " << end_msg << " (" << step << " steps, " << rejections << " rejected, "
            << wall_now() - wall_start << " s wall: thermal " << tm_thermal << " s, mechanics " << tm_mech
            << " s, metrics " << tm_metrics << " s, " << amg_builds << "/" << amg_builds_T << " AMG builds (mech/thermal), "
            << pcg_T_total << " thermal PCG iterations)\n";
  write_pvd();
  write_summary(complete, end_msg);
  if (hist) std::fclose(hist);
  hist = nullptr;
}

}  // namespace pcz
