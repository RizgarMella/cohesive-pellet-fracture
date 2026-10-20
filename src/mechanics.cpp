// Small-strain linear thermo-elasticity of the fuel with thermal and swelling
// eigenstrains, cohesive/contact interfaces and the platen contact. The
// nonlinear system is solved by a modified Newton method with the symmetric
// secant interface matrix (always positive definite), a backtracking line
// search, and an OpenMP PCG linear solver.
#include <algorithm>
#include <cmath>
#include <iostream>

#include "fe.hpp"
#include "model.hpp"

namespace pcz {

namespace {
inline void atomic_add(double& x, double v) {
#pragma omp atomic
  x += v;
}
}  // namespace

void Model::assemble_bulk_stiffness() {
  Kb.zero();
  const int ne = int(mesh.nelem());
  const double lam = mat.lambda(), mu = mat.mu();
#pragma omp parallel for schedule(static)
  for (int e = 0; e < ne; ++e) {
    double X[8][3];
    for (int a = 0; a < 8; ++a)
      for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[mesh.hex[e][a]][i];
    double Ke[8][8][9] = {};
    for (int g = 0; g < 8; ++g) {
      double N[8], B[8][3], dV;
      hex_gp(X, g, N, B, dV);
      for (int a = 0; a < 8; ++a)
        for (int b = 0; b < 8; ++b) {
          double dd = B[a][0] * B[b][0] + B[a][1] * B[b][1] + B[a][2] * B[b][2];
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              Ke[a][b][i * 3 + j] += (lam * B[a][i] * B[b][j] + mu * B[a][j] * B[b][i] + (i == j ? mu * dd : 0.0)) * dV;
        }
    }
    for (int a = 0; a < 8; ++a)
      for (int b = 0; b < 8; ++b) {
        double* blk = Kb.block(hex_blk[e][a * 8 + b]);
        for (int k = 0; k < 9; ++k) atomic_add(blk[k], Ke[a][b][k]);
      }
  }
}

void Model::compute_eigen_forces() {
  const int nn = int(mesh.nnode()), ne = int(mesh.nelem());
  F_eig.assign(size_t(3) * nn, 0.0);
  std::vector<double> F_dev(size_t(3) * nn, 0.0);
  const double c3 = 3 * mat.lambda() + 2 * mat.mu();
  const double sw_solid = mat.solid_swelling(fima_trial);
  // volume-mean eigenstrain: its (self-equilibrated) part is removed from the
  // force reference used by the Newton tolerance, so the tolerance measures
  // the stress-producing loads rather than the free uniform expansion
  double emean = 0, vtot = 0;
#pragma omp parallel for reduction(+ : emean, vtot)
  for (int e = 0; e < ne; ++e) {
    double Tc = 0;
    for (int a = 0; a < 8; ++a) Tc += T[mesh.hex[e][a]] / 8;
    emean += (mat.alpha * (Tc - mat.T_ref) + (sw_solid + gas_swell_trial[e]) / 3.0) * elem_vol[e];
    vtot += elem_vol[e];
  }
  emean /= vtot;
#pragma omp parallel for schedule(static)
  for (int e = 0; e < ne; ++e) {
    double X[8][3], Te[8];
    for (int a = 0; a < 8; ++a) {
      int n = mesh.hex[e][a];
      for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[n][i];
      Te[a] = T[n];
    }
    double fe[8][3] = {}, fd[8][3] = {};
    double sw = (sw_solid + gas_swell_trial[e]) / 3.0;
    for (int g = 0; g < 8; ++g) {
      double N[8], B[8][3], dV;
      hex_gp(X, g, N, B, dV);
      double Tg = 0;
      for (int a = 0; a < 8; ++a) Tg += N[a] * Te[a];
      double eps = mat.alpha * (Tg - mat.T_ref) + sw;
      for (int a = 0; a < 8; ++a)
        for (int i = 0; i < 3; ++i) {
          fe[a][i] += B[a][i] * c3 * eps * dV;
          fd[a][i] += B[a][i] * c3 * (eps - emean) * dV;
        }
    }
    for (int a = 0; a < 8; ++a)
      for (int i = 0; i < 3; ++i) {
        atomic_add(F_eig[3 * mesh.hex[e][a] + i], fe[a][i]);
        atomic_add(F_dev[3 * mesh.hex[e][a] + i], fd[a][i]);
      }
  }
  Fref_dev = norm2(F_dev);
}

// Evaluate interface (and platen) tractions at the current u. Adds internal
// forces to *F (if not null) and secant stiffness to Kmat (if assemble).
void Model::interface_forces(double dt, std::vector<double>* F, bool assemble) {
  const long ni = long(mesh.ifaces.size());
  double fimax = 0.0;
  int nsoft = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(max : fimax) reduction(+ : nsoft)
  for (long f = 0; f < ni; ++f) {
    const auto& ifc = mesh.ifaces[f];
    const bool cohesive = ifc.kind != IF_PELLET || mesh.p.cohesive_pellet_interface;
    double fA[4][3] = {}, Kloc[4][4][9] = {};
    for (int g = 0; g < 4; ++g) {
      size_t q = size_t(f) * 4 + g;
      double xi = QUAD_SIGN[g][0] * GP1, eta = QUAD_SIGN[g][1] * GP1;
      double N[4];
      for (int a = 0; a < 4; ++a) N[a] = 0.25 * (1 + QUAD_SIGN[a][0] * xi) * (1 + QUAD_SIGN[a][1] * eta);
      double jump[3] = {0, 0, 0};
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 3; ++i) jump[i] += N[a] * (u[3 * ifc.b[a] + i] - u[3 * ifc.a[a] + i]);
      const double* R = gp_R[q].data();
      double dl[3];
      for (int k = 0; k < 3; ++k) dl[k] = R[k * 3] * jump[0] + R[k * 3 + 1] * jump[1] + R[k * 3 + 2] * jump[2];
      LawOut o;
      if (cohesive)
        cohesive_law(coh, gp_props[q], gp_state[q], dl, dt, gp_trial[q], o);
      else
        contact_law(con, gp_props[q], gp_state[q], dl, gp_trial[q], o);
      gp_open[q] = float(dl[0] + gp_props[q].gap0);
      if (cohesive) {
        // failure index of intact points and number of points still softening
        const auto& tr = gp_trial[q];
        if (tr.d == 0.0) {
          double dnp = std::max(dl[0], 0.0);
          double de = std::sqrt(dnp * dnp + coh.beta * coh.beta * (dl[1] * dl[1] + dl[2] * dl[2]));
          fimax = std::max(fimax, de / delta0(coh, gp_props[q]));
        } else if (tr.d - tr.dv > 0.02 || tr.d - gp_state[q].d > 0.05) {
          ++nsoft;  // unstable growth: viscous damage lags, or a large damage jump in this step
        }
      }
      gp_tn[q] = float(o.t[0]);
      gp_slip[q] = o.slipping ? 1.f : 0.f;
      gp_ts[q] = float(std::hypot(o.t[1], o.t[2]));
      const double dA = gp_dA[q];
      // global traction  tg = R^T t
      double tg[3];
      for (int i = 0; i < 3; ++i) tg[i] = R[i] * o.t[0] + R[3 + i] * o.t[1] + R[6 + i] * o.t[2];
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 3; ++i) fA[a][i] += N[a] * tg[i] * dA;
      if (assemble) {
        double Kg[9];  // R^T D R
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k)
              for (int l = 0; l < 3; ++l) s += R[k * 3 + i] * o.D[k][l] * R[l * 3 + j];
            Kg[i * 3 + j] = s;
          }
        for (int a = 0; a < 4; ++a)
          for (int b = 0; b < 4; ++b) {
            double w = N[a] * N[b] * dA;
            for (int k = 0; k < 9; ++k) Kloc[a][b][k] += w * Kg[k];
          }
      }
    }
    if (F) {
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 3; ++i) {
          atomic_add((*F)[3 * ifc.b[a] + i], fA[a][i]);
          atomic_add((*F)[3 * ifc.a[a] + i], -fA[a][i]);
        }
    }
    if (assemble) {
      for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
          double* AA = Kmat.block(if_blk[f][a * 8 + b]);
          double* BB = Kmat.block(if_blk[f][(a + 4) * 8 + b + 4]);
          double* AB = Kmat.block(if_blk[f][a * 8 + b + 4]);
          double* BA = Kmat.block(if_blk[f][(a + 4) * 8 + b]);
          for (int k = 0; k < 9; ++k) {
            double v = Kloc[a][b][k];
            atomic_add(AA[k], v);
            atomic_add(BB[k], v);
            atomic_add(AB[k], -v);
            atomic_add(BA[k], -v);
          }
        }
    }
  }
  failure_index = fimax;
  n_softening = nsoft;
  // rigid platen under the bottom face (normal +z into the pellet)
  const long np = long(platen_node.size());
  ContactParams pc = con;
  pc.K = mbc.platen_stiffness;
  pc.friction = mbc.platen_friction;
#pragma omp parallel for schedule(static)
  for (long i = 0; i < np; ++i) {
    int n = platen_node[i];
    double dl[3] = {u[3 * n + 2], u[3 * n], u[3 * n + 1]};  // (n, t1, t2) = (z, x, y)
    GPProps gpp;
    LawOut o;
    contact_law(pc, gpp, platen_state[i], dl, platen_trial[i], o);
    double A = platen_area[i];
    if (F) {
      atomic_add((*F)[3 * n + 2], o.t[0] * A);
      atomic_add((*F)[3 * n], o.t[1] * A);
      atomic_add((*F)[3 * n + 1], o.t[2] * A);
    }
    if (assemble) {
      double* blk = Kmat.block(Kmat.diag[n]);
      atomic_add(blk[8], o.D[0][0] * A);
      atomic_add(blk[0], o.D[1][1] * A);
      atomic_add(blk[4], o.D[2][2] * A);
    }
  }
}

bool Model::solve_mechanics(double t, double dt, Diagnostics& dg) {
  const int nn = int(mesh.nnode());
  const long N = 3L * nn;
  // prescribed displacements (all masked dofs are zero except the pressed top face)
  for (long i = 0; i < N; ++i)
    if (mask_u[i] == 0.0) u[i] = 0.0;
  if (mbc.top == "displacement") {
    double w = mbc.top_displacement(t);
    for (int n : top_nodes) u[3 * n + 2] = w;
  }
  // external loads
  compute_eigen_forces();
  F_ext = F_eig;
  double Fp = 0;
  if (mbc.top == "pressure") {
    double p = mbc.top_pressure(t);
    for (auto& bf : mesh.top) {
      double X[4][3];
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[bf.n[a]][i];
      for (int g = 0; g < 4; ++g) {
        double Nq[4], dA, R[3][3], x[3];
        quad_gp(X, g, Nq, dA, R, x);
        for (int a = 0; a < 4; ++a)
          for (int i = 0; i < 3; ++i) F_ext[3 * bf.n[a] + i] -= p * R[0][i] * Nq[a] * dA;
      }
    }
    Fp = std::abs(p) * M_PI * mesh.p.R * mesh.p.R;
  }
  Fref = std::max(Fref_dev, Fp);
  if (Fref <= 0) Fref = 1.0;
  // mean diagonal for the stabilisation shift
  double diag_mean = 0;
  for (int n = 0; n < nn; ++n) {
    const double* b = &Kb.val[size_t(Kb.diag[n]) * 9];
    diag_mean += b[0] + b[4] + b[8];
  }
  diag_mean /= (3.0 * nn);

  std::vector<double> R(N), Fi(N), du(N), u0(N);
  auto residual = [&](bool assemble) {
    Kb.matvec(u, R);
    std::fill(Fi.begin(), Fi.end(), 0.0);
    if (assemble) Kmat.val = Kb.val;
    interface_forces(dt, &Fi, assemble);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i) R[i] = F_ext[i] - R[i] - Fi[i];
    double s = 0;
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (long i = 0; i < N; ++i) s += R[i] * R[i] * mask_u[i];
    return std::sqrt(s);
  };

  dg.newton_iters = 0;
  dg.lin_iters = 0;
  double rn = residual(true);
  std::vector<double> rhist;
  for (int it = 0; it < sp.newton_max; ++it) {
    dg.res = rn / Fref;
    if (!std::isfinite(rn)) return false;
    rhist.push_back(rn);
    // stagnation: less than 2x reduction over the last 8 iterations
    if (it >= 12 && rn > 0.5 * rhist[it - 8]) return false;
    if (rn <= 1e-3 * sp.newton_tol * Fref) {  // already (essentially) in equilibrium
      Rres = R;
      return true;
    }
    if (sp.stabilization > 0) Kmat.add_diagonal(sp.stabilization * diag_mean);
    std::fill(du.begin(), du.end(), 0.0);
    // rebuild the hierarchy only when the stale one has clearly degraded
    const bool rebuild = !amg_u.ready() || (last_pcg_u > sp.amg_rebuild_iters && last_pcg_u > 2 * base_pcg_u);
    if (sp.amg && rebuild) {
      amg_u.setup(Kmat, mask_u, &mesh.X);
      ++amg_builds;
    }
    auto lr = pcg(Kmat, R, du, mask_u, sp.lin_rtol, sp.lin_maxit, sp.amg ? &amg_u : nullptr);
    last_pcg_u = lr.iters;
    if (rebuild) base_pcg_u = std::max(lr.iters, 1);
    dg.lin_iters += lr.iters;
    ++dg.newton_iters;
    if (!lr.converged && lr.rel_res > 1e-3) {
      std::cerr << "  linear solver stalled (rel. residual " << lr.rel_res << ")\n";
      return false;
    }
    u0 = u;
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i) u[i] = u0[i] + du[i];
    double rfull = residual(true), abest = 1.0;
    if (sp.line_search && !(rfull < rn)) {
      // backtracking: keep the best of alpha = 1, 1/2, 1/4, 1/8
      double rbest = rfull, alpha = 1.0;
      for (int ls = 0; ls < 3; ++ls) {
        alpha *= 0.5;
#pragma omp parallel for schedule(static)
        for (long i = 0; i < N; ++i) u[i] = u0[i] + alpha * du[i];
        double r = residual(false);
        if (r < rbest) {
          rbest = r;
          abest = alpha;
        }
        if (r < rn) break;
      }
#pragma omp parallel for schedule(static)
      for (long i = 0; i < N; ++i) u[i] = u0[i] + abest * du[i];
      rfull = residual(true);
    }
    double dun = norm_inf(du) * abest, umax = norm_inf(u);
    rn = rfull;
    if (rn <= sp.newton_tol * Fref ||
        (dun <= sp.newton_dtol * std::max(umax, 1e-12) && rn <= 10 * sp.newton_tol * Fref)) {
      dg.res = rn / Fref;
      Rres = R;
      return true;
    }
  }
  dg.res = rn / Fref;
  return false;
}

void Model::element_stress(int e, double sig[6]) const {
  double X[8][3], ue[8][3], Te[8];
  for (int a = 0; a < 8; ++a) {
    int n = mesh.hex[e][a];
    for (int i = 0; i < 3; ++i) {
      X[a][i] = mesh.X[n][i];
      ue[a][i] = u[3 * n + i];
    }
    Te[a] = T[n];
  }
  const double lam = mat.lambda(), mu = mat.mu();
  const double sw = (mat.solid_swelling(fima) + gas_swell[e]) / 3.0;
  for (int k = 0; k < 6; ++k) sig[k] = 0;
  double vol = 0;
  for (int g = 0; g < 8; ++g) {
    double N[8], B[8][3], dV;
    hex_gp(X, g, N, B, dV);
    double H[3][3] = {};
    double Tg = 0;
    for (int a = 0; a < 8; ++a) {
      Tg += N[a] * Te[a];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) H[i][j] += ue[a][i] * B[a][j];
    }
    double ee = mat.alpha * (Tg - mat.T_ref) + sw;
    double ex = H[0][0] - ee, ey = H[1][1] - ee, ez = H[2][2] - ee;
    double tr = ex + ey + ez;
    sig[0] += (lam * tr + 2 * mu * ex) * dV;
    sig[1] += (lam * tr + 2 * mu * ey) * dV;
    sig[2] += (lam * tr + 2 * mu * ez) * dV;
    sig[3] += mu * (H[0][1] + H[1][0]) * dV;
    sig[4] += mu * (H[1][2] + H[2][1]) * dV;
    sig[5] += mu * (H[0][2] + H[2][0]) * dV;
    vol += dV;
  }
  for (int k = 0; k < 6; ++k) sig[k] /= vol;
}

}  // namespace pcz
