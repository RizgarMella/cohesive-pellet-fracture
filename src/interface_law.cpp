#include "interface_law.hpp"

#include <algorithm>
#include <cmath>

namespace pcz {

double delta0(const CohesiveParams& P, const GPProps& g) { return g.sigma_t / P.K; }

double deltaf(const CohesiveParams& P, const GPProps& g) {
  if (P.softening == 0) return 2.0 * g.Gc / g.sigma_t;
  // exponential: T = sigma_t exp(-(de-d0)/dc), Gc = sigma_t d0/2 + sigma_t dc
  return (g.Gc - 0.5 * g.sigma_t * delta0(P, g)) / g.sigma_t;
}

double min_fracture_energy(const CohesiveParams& P, double sigma_t) {
  return 0.5 * sigma_t * sigma_t / P.K;  // Gc must exceed the elastic energy at peak
}

double damage_from_separation(const CohesiveParams& P, const GPProps& g, double de) {
  const double d0 = delta0(P, g);
  if (de <= d0) return 0.0;
  if (P.softening == 0) {
    const double df = deltaf(P, g);
    if (de >= df) return 1.0;
    return df * (de - d0) / (de * (df - d0));
  }
  const double dc = deltaf(P, g);
  double T = g.sigma_t * std::exp(-(de - d0) / dc);
  return std::min(1.0, std::max(0.0, 1.0 - T / (P.K * de)));
}

double dissipated_energy(const CohesiveParams& P, const GPProps& g, double dm) {
  const double d0 = delta0(P, g);
  if (dm <= d0) return 0.0;
  double area, Tm;
  if (P.softening == 0) {
    const double df = deltaf(P, g);
    if (dm >= df) return g.Gc;
    Tm = g.sigma_t * (df - dm) / (df - d0);
    area = 0.5 * g.sigma_t * d0 + 0.5 * (g.sigma_t + Tm) * (dm - d0);
  } else {
    const double dc = deltaf(P, g);
    Tm = g.sigma_t * std::exp(-(dm - d0) / dc);
    area = 0.5 * g.sigma_t * d0 + g.sigma_t * dc * (1.0 - std::exp(-(dm - d0) / dc));
  }
  return std::max(0.0, area - 0.5 * Tm * dm);  // minus recoverable elastic energy
}

static void friction_update(double Kt, double mu, double pn, double ds1, double ds2, double s1, double s2,
                            double& t1, double& t2, double& ns1, double& ns2, bool& slipping) {
  double tr1 = Kt * (ds1 - s1), tr2 = Kt * (ds2 - s2);
  double tn = std::sqrt(tr1 * tr1 + tr2 * tr2);
  double lim = (mu < 0) ? 1e300 : mu * pn;
  if (tn <= lim) {
    t1 = tr1;
    t2 = tr2;
    ns1 = s1;
    ns2 = s2;
    slipping = false;
  } else {
    double f = lim / tn;
    t1 = tr1 * f;
    t2 = tr2 * f;
    ns1 = ds1 - t1 / Kt;
    ns2 = ds2 - t2 / Kt;
    slipping = true;
  }
}

void cohesive_law(const CohesiveParams& P, const GPProps& g, const GPState& st, const double dl[3], double dt,
                  GPState& tr, LawOut& o) {
  const double K = P.K, Ks = P.K * P.Ks_ratio;
  const double dn = dl[0], ds1 = dl[1], ds2 = dl[2];
  const double dnp = std::max(dn, 0.0);
  const double de = std::sqrt(dnp * dnp + P.beta * P.beta * (ds1 * ds1 + ds2 * ds2));
  tr = st;
  tr.dmax = std::max(st.dmax, de);
  tr.d = std::max(st.d, damage_from_separation(P, g, tr.dmax));
  if (P.viscosity > 0 && dt > 0) {
    double r = dt / P.viscosity;
    tr.dv = (st.dv + r * tr.d) / (1.0 + r);
  } else {
    tr.dv = tr.d;
  }
  tr.dv = std::max(tr.dv, st.dv);
  const double w = 1.0 - tr.dv;            // stiffness fraction in the tractions
  const double wm = std::max(w, P.residual);  // matrix fraction (never singular)
  for (auto& r : o.D) r[0] = r[1] = r[2] = 0.0;
  o.slipping = false;
  if (dn >= 0) {
    o.t[0] = w * K * dn;
    o.D[0][0] = wm * K;
    o.open = tr.dv > 0;
    o.contact = false;
    // no friction on open faces: forget slip history (re-contact starts stress free)
    double t1 = w * Ks * ds1, t2 = w * Ks * ds2;
    o.t[1] = t1;
    o.t[2] = t2;
    o.D[1][1] = o.D[2][2] = wm * Ks;
    tr.s1 = ds1;
    tr.s2 = ds2;
  } else {
    o.t[0] = K * dn;  // compression: full penalty
    o.D[0][0] = K;
    o.open = false;
    o.contact = true;
    double ft1 = 0, ft2 = 0, ns1 = ds1, ns2 = ds2;
    if (P.friction != 0.0 && tr.dv > 0) {
      friction_update(Ks, P.friction, -o.t[0], ds1, ds2, st.s1, st.s2, ft1, ft2, ns1, ns2, o.slipping);
    }
    tr.s1 = ns1;
    tr.s2 = ns2;
    o.t[1] = w * Ks * ds1 + tr.dv * ft1;
    o.t[2] = w * Ks * ds2 + tr.dv * ft2;
    double kt = w * Ks + (P.friction != 0.0 ? tr.dv * Ks : 0.0);
    o.D[1][1] = o.D[2][2] = std::max(kt, P.residual * Ks);
  }
}

void contact_law(const ContactParams& P, const GPProps& g, const GPState& st, const double dl[3], GPState& tr,
                 LawOut& o) {
  const double K = P.K, Kt = P.K * P.Kt_ratio;
  const double gap = g.gap0 + dl[0];
  tr = st;
  for (auto& r : o.D) r[0] = r[1] = r[2] = 0.0;
  o.slipping = false;
  if (gap >= 0) {
    o.t[0] = o.t[1] = o.t[2] = 0.0;
    o.D[0][0] = P.residual * K;
    o.D[1][1] = o.D[2][2] = P.residual * Kt;
    o.open = true;
    o.contact = false;
    tr.s1 = dl[1];
    tr.s2 = dl[2];
    return;
  }
  o.open = false;
  o.contact = true;
  o.t[0] = K * gap;
  o.D[0][0] = K;
  double t1 = 0, t2 = 0, ns1 = dl[1], ns2 = dl[2];
  if (P.friction != 0.0) {
    friction_update(Kt, P.friction, -o.t[0], dl[1], dl[2], st.s1, st.s2, t1, t2, ns1, ns2, o.slipping);
    o.D[1][1] = o.D[2][2] = Kt;
  } else {
    o.D[1][1] = o.D[2][2] = P.residual * Kt;
  }
  tr.s1 = ns1;
  tr.s2 = ns2;
  o.t[1] = t1;
  o.t[2] = t2;
}

}  // namespace pcz
