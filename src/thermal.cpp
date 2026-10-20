// Transient heat conduction with temperature/burnup dependent conductivity,
// uniform volumetric heating, convective cooling of the pellet outer surface
// and conductance across interfaces: perfect (bonded) for intact cohesive
// surfaces, helium gap conduction k_g/(d_g + roughness) plus linearised
// radiation once cracked, and solid + gas conductance at pellet contacts.
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "fe.hpp"
#include "model.hpp"

namespace pcz {

namespace {
inline void atomic_add(double& x, double v) {
#pragma omp atomic
  x += v;
}
}  // namespace

bool Model::solve_thermal(double t, double dt, int& iters) {
  const int nn = int(mesh.nnode()), ne = int(mesh.nelem());
  const double q3 = power_density(t);
  const double Tcool = tbc.T_coolant(t);
  const double eps_eff = 1.0 / (2.0 / tbc.emissivity - 1.0);
  std::vector<double> rhs(nn), res(nn), dT(nn);
  if (tbc.fixed_surface)
    for (auto& bf : mesh.outer)
      for (int a = 0; a < 4; ++a) T[bf.n[a]] = Tcool;

  iters = 0;
  for (int it = 0; it < sp.thermal_max; ++it) {
    Kth.zero();
    std::fill(rhs.begin(), rhs.end(), 0.0);
    // bulk conduction, capacity, heating
#pragma omp parallel for schedule(static)
    for (int e = 0; e < ne; ++e) {
      double X[8][3], Te[8], To[8];
      for (int a = 0; a < 8; ++a) {
        int n = mesh.hex[e][a];
        for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[n][i];
        Te[a] = T[n];
        To[a] = T_old[n];
      }
      double Ke[8][8] = {};
      double Ce[8] = {}, Qe[8] = {};
      for (int g = 0; g < 8; ++g) {
        double N[8], B[8][3], dV;
        hex_gp(X, g, N, B, dV);
        double Tg = 0;
        for (int a = 0; a < 8; ++a) Tg += N[a] * Te[a];
        double k = mat.k(Tg, fima_trial);
        double rc = mat.density * mat.cp(Tg);
        for (int a = 0; a < 8; ++a) {
          for (int b = a; b < 8; ++b)
            Ke[a][b] += k * (B[a][0] * B[b][0] + B[a][1] * B[b][1] + B[a][2] * B[b][2]) * dV;
          Ce[a] += rc * N[a] * dV;
          Qe[a] += q3 * N[a] * dV;
        }
      }
      for (int a = 0; a < 8; ++a)
        for (int b = 0; b < a; ++b) Ke[a][b] = Ke[b][a];
      for (int a = 0; a < 8; ++a) {
        if (dt > 0) Ke[a][a] += Ce[a] / dt;
        for (int b = 0; b < 8; ++b) atomic_add(Kth.val[hex_blk[e][a * 8 + b]], Ke[a][b]);
        atomic_add(rhs[mesh.hex[e][a]], Qe[a] + (dt > 0 ? Ce[a] / dt * To[a] : 0.0));
      }
    }
    // convection on the outer surface
    if (!tbc.fixed_surface) {
#pragma omp parallel for schedule(static)
      for (long f = 0; f < long(mesh.outer.size()); ++f) {
        const auto& bf = mesh.outer[f];
        double X[4][3];
        for (int a = 0; a < 4; ++a)
          for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[bf.n[a]][i];
        for (int g = 0; g < 4; ++g) {
          double N[4], dA, R[3][3], x[3];
          quad_gp(X, g, N, dA, R, x);
          for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
              int k = Kth.find(bf.n[a], bf.n[b]);
              atomic_add(Kth.val[k], tbc.h_conv * N[a] * N[b] * dA);
            }
            atomic_add(rhs[bf.n[a]], tbc.h_conv * Tcool * N[a] * dA);
          }
        }
      }
    }
    // interfaces
    const long ni = long(mesh.ifaces.size());
#pragma omp parallel for schedule(static)
    for (long f = 0; f < ni; ++f) {
      const auto& ifc = mesh.ifaces[f];
      const bool cohesive = ifc.kind != IF_PELLET || mesh.p.cohesive_pellet_interface;
      for (int g = 0; g < 4; ++g) {
        size_t q = size_t(f) * 4 + g;
        double xi = QUAD_SIGN[g][0] * GP1, eta = QUAD_SIGN[g][1] * GP1;
        double N[4];
        for (int a = 0; a < 4; ++a) N[a] = 0.25 * (1 + QUAD_SIGN[a][0] * xi) * (1 + QUAD_SIGN[a][1] * eta);
        double TA = 0, TB = 0, jump[3] = {0, 0, 0};
        for (int a = 0; a < 4; ++a) {
          TA += N[a] * T[ifc.a[a]];
          TB += N[a] * T[ifc.b[a]];
          for (int i = 0; i < 3; ++i) jump[i] += N[a] * (u[3 * ifc.b[a] + i] - u[3 * ifc.a[a] + i]);
        }
        const auto& Rr = gp_R[q];
        double dn = Rr[0] * jump[0] + Rr[1] * jump[1] + Rr[2] * jump[2];
        double Tm = 0.5 * (TA + TB);
        double hrad = STEFAN_BOLTZMANN * eps_eff * (TA * TA + TB * TB) * (TA + TB);
        double h;
        if (cohesive) {
          double d = gp_trial[q].dv;
          double open = std::max(dn, 0.0);
          h = (1.0 - d) * tbc.h_bonded + d * (helium_conductivity(Tm) / (open + tbc.roughness) + hrad);
        } else {
          double gap = gp_props[q].gap0 + dn;
          h = helium_conductivity(Tm) / (std::max(gap, 0.0) + tbc.roughness) + hrad;
          if (gap < 0) h += tbc.h_contact;
        }
        // lumped (nodal-quadrature) conductance: couples only paired nodes, so the
        // matrix stays an M-matrix even for very large bonded conductance
        double dA = gp_dA[q];
        for (int a = 0; a < 4; ++a) {
          double v = h * N[a] * dA;
          atomic_add(Kth.val[if_blk[f][a * 8 + a]], v);
          atomic_add(Kth.val[if_blk[f][(a + 4) * 8 + a + 4]], v);
          atomic_add(Kth.val[if_blk[f][a * 8 + a + 4]], -v);
          atomic_add(Kth.val[if_blk[f][(a + 4) * 8 + a]], -v);
        }
      }
    }
    // residual and Picard / Newton-like increment
    Kth.matvec(T, res);
#pragma omp parallel for
    for (int n = 0; n < nn; ++n) res[n] = rhs[n] - res[n];
    std::fill(dT.begin(), dT.end(), 0.0);
    const bool rebuild = !amg_T.ready() || (last_pcg_T > sp.amg_rebuild_iters && last_pcg_T > 2 * base_pcg_T);
    if (sp.amg && rebuild) {
      amg_T.prm.theta = 0.08;  // keeps bonded-interface pairs from swamping the aggregation
      amg_T.setup(Kth, mask_T, nullptr);
      ++amg_builds_T;
    }
    auto r = pcg(Kth, res, dT, mask_T, 1e-8, sp.lin_maxit, sp.amg ? &amg_T : nullptr);
    last_pcg_T = r.iters;
    if (rebuild) base_pcg_T = std::max(r.iters, 1);
    pcg_T_total += r.iters;
    double m = norm_inf(dT);
#pragma omp parallel for
    for (int n = 0; n < nn; ++n) T[n] += dT[n];
    ++iters;
    if (!r.converged && r.rel_res > 1e-6) return false;
    if (!std::isfinite(m)) return false;
    if (m < sp.thermal_tol) return true;
  }
  return false;
}

}  // namespace pcz
