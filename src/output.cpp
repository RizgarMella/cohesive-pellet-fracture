// Run-directory outputs (self-describing, units in column names):
//   manifest.json        inputs, derived quantities, build id, labels
//   history.csv          one row per accepted step
//   bulk_NNNN.vtu        hex mesh: displacement, temperature, stresses (ParaView)
//   surface_NNNN.vtu     boundary quads (outer/top/bottom) with parent-element stress
//   cracks_NNNN.vtu      interface quads: damage, opening, strength, toughness, traction
//   angular_NNNN.csv     damage vs angle on pellet end faces / mid-height (alignment)
//   interface_NNNN.csv   pellet-pellet contact state per integration point
//   fragments.csv        fragments (regions joined by intact ligaments) at the end
//   summary.json         written last; "complete": true only for finished runs
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "fe.hpp"
#include "model.hpp"

#ifndef PELLETCZ_BUILD_ID
#define PELLETCZ_BUILD_ID "unknown"
#endif

namespace pcz {

std::string build_identifier() { return PELLETCZ_BUILD_ID; }

namespace {

std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (c == '\n') {
      o += "\\n";
      continue;
    }
    o += c;
  }
  return o + "\"";
}

std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  std::ostringstream o;
  o << std::setprecision(10) << v;
  return o.str();
}

// base64 encoder for VTK binary arrays
std::string b64(const unsigned char* d, size_t n) {
  static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o;
  o.reserve(4 * ((n + 2) / 3));
  size_t i = 0;
  for (; i + 2 < n; i += 3) {
    unsigned v = (d[i] << 16) | (d[i + 1] << 8) | d[i + 2];
    o += tbl[(v >> 18) & 63];
    o += tbl[(v >> 12) & 63];
    o += tbl[(v >> 6) & 63];
    o += tbl[v & 63];
  }
  if (i < n) {
    unsigned v = d[i] << 16;
    if (i + 1 < n) v |= d[i + 1] << 8;
    o += tbl[(v >> 18) & 63];
    o += tbl[(v >> 12) & 63];
    o += (i + 1 < n) ? tbl[(v >> 6) & 63] : '=';
    o += '=';
  }
  return o;
}

struct VtuWriter {
  std::ofstream f;
  bool binary;
  VtuWriter(const std::string& path, bool bin, size_t npts, size_t ncells) : f(path), binary(bin) {
    f << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" "
         "header_type=\"UInt64\">\n<UnstructuredGrid>\n<Piece NumberOfPoints=\""
      << npts << "\" NumberOfCells=\"" << ncells << "\">\n";
  }
  template <class Tp>
  void array(const char* name, const char* type, int ncomp, const std::vector<Tp>& v) {
    f << "<DataArray type=\"" << type << "\" Name=\"" << name << "\" NumberOfComponents=\"" << ncomp
      << "\" format=\"" << (binary ? "binary" : "ascii") << "\">\n";
    if (binary) {
      uint64_t nb = v.size() * sizeof(Tp);
      std::vector<unsigned char> buf(8 + nb);
      std::memcpy(buf.data(), &nb, 8);
      if (nb) std::memcpy(buf.data() + 8, v.data(), nb);
      f << b64(buf.data(), buf.size()) << "\n";
    } else {
      f << std::setprecision(8);
      for (size_t i = 0; i < v.size(); ++i) f << v[i] << ((i + 1) % 12 ? ' ' : '\n');
      f << "\n";
    }
    f << "</DataArray>\n";
  }
  void close() { f << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n"; }
};

std::string idx4(int i) {
  std::ostringstream o;
  o << std::setw(4) << std::setfill('0') << i;
  return o.str();
}

}  // namespace

void Model::write_manifest() const {
  std::ofstream f(op.dir + "/manifest.json");
  // derived quantities
  double sum_s = 0, sum_g = 0, min_s = 1e300, max_s = 0;
  size_t nc = 0;
  for (size_t f2 = 0; f2 < mesh.ifaces.size(); ++f2) {
    if (mesh.ifaces[f2].kind == IF_PELLET && !mesh.p.cohesive_pellet_interface) continue;
    for (int g = 0; g < 4; ++g) {
      const auto& P = gp_props[f2 * 4 + g];
      sum_s += P.sigma_t;
      sum_g += P.Gc;
      min_s = std::min(min_s, P.sigma_t);
      max_s = std::max(max_s, P.sigma_t);
      ++nc;
    }
  }
  double ms = nc ? sum_s / nc : strength.mean, mg = nc ? sum_g / nc : toughness.mean;
  GPProps gm;
  gm.sigma_t = strength.mean;
  gm.Gc = toughness.mean;
  double lcz = mat.E * toughness.mean / (strength.mean * strength.mean);  // Turon et al. (2007) estimate
  double dr_min = 1e300, dr_max = 0;
  for (size_t i = 1; i < mesh.r_nodes.size(); ++i) {
    dr_min = std::min(dr_min, mesh.r_nodes[i] - mesh.r_nodes[i - 1]);
    dr_max = std::max(dr_max, mesh.r_nodes[i] - mesh.r_nodes[i - 1]);
  }
  double dz = mesh.p.H / mesh.p.nz;
  double dth = 2 * M_PI * mesh.p.R / mesh.p.ntheta;
  double h_out = std::max({mesh.r_nodes.back() - mesh.r_nodes[mesh.r_nodes.size() - 2], dz, dth});
  double l_ch = mat.E * toughness.mean / (strength.mean * strength.mean);
  f << "{\n \"solver\": \"pelletcz\",\n \"build_id\": " << jstr(build_identifier())
    << ",\n \"input_file\": " << jstr(cfg.source()) << ",\n \"labels\": {";
  std::string lab = cfg.get_string("output.labels", "");
  {
    std::stringstream ss(lab);
    std::string kv;
    bool first = true;
    while (std::getline(ss, kv, ';')) {
      auto eq = kv.find('=');
      if (eq == std::string::npos) continue;
      f << (first ? "" : ", ") << jstr(trim(kv.substr(0, eq))) << ": " << jstr(trim(kv.substr(eq + 1)));
      first = false;
    }
  }
  f << "},\n \"config\": {";
  {
    std::stringstream ss(cfg.dump());
    std::string line;
    bool first = true;
    while (std::getline(ss, line)) {
      auto eq = line.find(" = ");
      if (eq == std::string::npos) continue;
      f << (first ? "\n  " : ",\n  ") << jstr(line.substr(0, eq)) << ": " << jstr(line.substr(eq + 3));
      first = false;
    }
  }
  f << "\n },\n \"derived\": {\n"
    << "  \"n_nodes\": " << mesh.nnode() << ",\n  \"n_elements\": " << mesh.nelem()
    << ",\n  \"n_interface_elements\": " << mesh.ifaces.size() << ",\n  \"n_cohesive_points\": " << nc
    << ",\n  \"dr_min_m\": " << jnum(dr_min) << ",\n  \"dr_max_m\": " << jnum(dr_max) << ",\n  \"dz_m\": " << jnum(dz)
    << ",\n  \"r_dtheta_outer_m\": " << jnum(dth) << ",\n  \"h_outer_max_m\": " << jnum(h_out)
    << ",\n  \"cohesive_zone_length_m\": " << jnum(lcz)
    << ",\n  \"zeta_lcz_over_h\": " << jnum(lcz / h_out)
    << ",\n  \"irwin_length_m\": " << jnum(l_ch) << ",\n  \"delta0_mean_m\": " << jnum(delta0(coh, gm))
    << ",\n  \"deltaf_mean_m\": " << jnum(deltaf(coh, gm)) << ",\n  \"sigma_t_sample_mean_Pa\": " << jnum(ms)
    << ",\n  \"sigma_t_sample_min_Pa\": " << jnum(min_s) << ",\n  \"sigma_t_sample_max_Pa\": " << jnum(max_s)
    << ",\n  \"Gc_sample_mean_Jpm2\": " << jnum(mg) << ",\n  \"Gc_clamped_points\": " << n_gc_clamped
    << ",\n  \"pellet_volume_m3\": " << jnum(M_PI * mesh.p.R * mesh.p.R * mesh.p.H) << "\n }\n}\n";
  std::ofstream(op.dir + "/input_used.ini") << cfg.dump();
}

void Model::open_history() {
  hist = std::fopen((op.dir + "/history.csv").c_str(), "w");
  std::fprintf(hist,
               "step,time_s,dt_s,linear_power_Wpm,burnup_MWd_kgU,fima,T_centre_K,T_surface_K,dT_K,T_max_K,"
               "crack_area_mm2,crack_area_radial_mm2,crack_area_transverse_mm2,crack_area_circumferential_mm2,"
               "crack_area_pellet_interface_mm2,damaged_area_mm2,dissipated_energy_mJ,n_fragments,max_principal_MPa,"
               "if_contact_force_N,if_contact_area_mm2,if_mean_pressure_MPa,if_max_pressure_MPa,if_slip_fraction,"
               "if_shear_force_N,top_pressure_MPa,top_reaction_N,bottom_reaction_N,pin_reaction_N");
  for (int p = 0; p < mesh.p.n_pellets; ++p) std::fprintf(hist, ",crack_area_p%d_mm2", p);
  std::fprintf(hist, ",newton_iters,pcg_iters,thermal_iters,residual,wall_s\n");
}

void Model::write_history_row(const Diagnostics& dg) {
  using namespace std::chrono;
  double wall = duration<double>(steady_clock::now().time_since_epoch()).count() - wall_start;
  const double NU_mass = mat.density * U_MASS_FRACTION;  // kgU/m3
  double bu = fima * ENERGY_PER_FISSION * (NU_mass / 0.238 * AVOGADRO) / NU_mass / 8.64e10;
  double area = met.area[0] + met.area[1] + met.area[2];
  double p_top = (mbc.top == "pressure") ? mbc.top_pressure(time) : 0.0;
  std::fprintf(hist, "%d,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%.6g,"
                     "%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g",
               step, time, step ? time - last["time_s"] : 0.0, linear_power(time), bu, fima, met.Tc, met.Ts,
               met.Tc - met.Ts, met.Tmax, area * 1e6, met.area[0] * 1e6, met.area[1] * 1e6, met.area[2] * 1e6,
               met.area[3] * 1e6, met.damaged_area * 1e6, met.dissipated * 1e3, met.n_fragments,
               met.max_principal * 1e-6, met.if_force, met.if_area * 1e6,
               met.if_area > 0 ? met.if_force / met.if_area * 1e-6 : 0.0, met.if_pmax * 1e-6, met.if_slip_frac,
               met.if_shear_force, p_top * 1e-6, met.top_reaction, met.bottom_reaction, met.pin_reaction);
  for (double a : met.pellet_area) std::fprintf(hist, ",%.6g", a * 1e6);
  std::fprintf(hist, ",%d,%d,%d,%.4g,%.4g\n", dg.newton_iters, dg.lin_iters, dg.thermal_iters, dg.res, wall);
  std::fflush(hist);
  last["time_s"] = time;
  last["burnup_MWd_kgU"] = bu;
  last["crack_area_mm2"] = area * 1e6;
  last["dissipated_energy_mJ"] = met.dissipated * 1e3;
  last["dT_K"] = met.Tc - met.Ts;
  last["T_centre_K"] = met.Tc;
  last["linear_power_Wpm"] = linear_power(time);
  last["max_principal_MPa"] = met.max_principal * 1e-6;
  last["n_fragments"] = met.n_fragments;
  last["wall_s"] = wall;
}

void Model::write_vtu(int idx) {
  const std::string s = idx4(idx);
  const int nn = int(mesh.nnode()), ne = int(mesh.nelem());
  std::vector<int> lab, erid;
  label_fragments(lab, erid);
  // element stresses
  std::vector<double> sig6(size_t(ne) * 6), smax(ne), shoop(ne), srr(ne), szz(ne), svm(ne);
#pragma omp parallel for
  for (int e = 0; e < ne; ++e) {
    double sg[6], ev[3];
    element_stress(e, sg);
    sym_eig3(sg, ev);
    for (int k = 0; k < 6; ++k) sig6[size_t(e) * 6 + k] = sg[k] * 1e-6;
    double c[3] = {0, 0, 0};
    for (int a = 0; a < 8; ++a)
      for (int i = 0; i < 3; ++i) c[i] += mesh.X[mesh.hex[e][a]][i] / 8;
    double th = std::atan2(c[1], c[0]), cs = std::cos(th), sn = std::sin(th);
    srr[e] = (sg[0] * cs * cs + sg[1] * sn * sn + 2 * sg[3] * sn * cs) * 1e-6;
    shoop[e] = (sg[0] * sn * sn + sg[1] * cs * cs - 2 * sg[3] * sn * cs) * 1e-6;
    szz[e] = sg[2] * 1e-6;
    smax[e] = ev[0] * 1e-6;
    svm[e] = std::sqrt(0.5 * ((ev[0] - ev[1]) * (ev[0] - ev[1]) + (ev[1] - ev[2]) * (ev[1] - ev[2]) +
                              (ev[2] - ev[0]) * (ev[2] - ev[0]))) * 1e-6;
  }
  std::vector<int> frag(ne);
  for (int e = 0; e < ne; ++e) frag[e] = lab[erid[e]];
  {
    VtuWriter w(op.dir + "/bulk_" + s + ".vtu", op.binary, nn, ne);
    std::vector<double> pts(size_t(3) * nn);
    for (int n = 0; n < nn; ++n)
      for (int i = 0; i < 3; ++i) pts[3 * n + i] = mesh.X[n][i];
    w.f << "<Points>\n";
    w.array("Points", "Float64", 3, pts);
    w.f << "</Points>\n<Cells>\n";
    std::vector<int32_t> conn(size_t(8) * ne), offs(ne);
    std::vector<uint8_t> types(ne, 12);
    for (int e = 0; e < ne; ++e) {
      for (int a = 0; a < 8; ++a) conn[size_t(8) * e + a] = mesh.hex[e][a];
      offs[e] = 8 * (e + 1);
    }
    w.array("connectivity", "Int32", 1, conn);
    w.array("offsets", "Int32", 1, offs);
    w.array("types", "UInt8", 1, types);
    w.f << "</Cells>\n<PointData Vectors=\"displacement_m\" Scalars=\"temperature_K\">\n";
    w.array("displacement_m", "Float64", 3, u);
    w.array("temperature_K", "Float64", 1, T);
    w.f << "</PointData>\n<CellData Scalars=\"max_principal_MPa\">\n";
    w.array("max_principal_MPa", "Float64", 1, smax);
    w.array("hoop_stress_MPa", "Float64", 1, shoop);
    w.array("radial_stress_MPa", "Float64", 1, srr);
    w.array("axial_stress_MPa", "Float64", 1, szz);
    w.array("von_mises_MPa", "Float64", 1, svm);
    w.array("stress_MPa", "Float64", 6, sig6);
    w.array("fragment", "Int32", 1, frag);
    std::vector<int32_t> pel(mesh.elem_pellet.begin(), mesh.elem_pellet.end());
    w.array("pellet", "Int32", 1, pel);
    w.f << "</CellData>\n";
    w.close();
  }
  // boundary surface (for light-weight rendering)
  {
    std::vector<const BFace*> faces;
    for (auto& b : mesh.outer) faces.push_back(&b);
    for (auto& b : mesh.top) faces.push_back(&b);
    for (auto& b : mesh.bottom) faces.push_back(&b);
    const size_t nf = faces.size();
    VtuWriter w(op.dir + "/surface_" + s + ".vtu", op.binary, 4 * nf, nf);
    std::vector<double> pts(12 * nf), disp(12 * nf), tt(4 * nf), sm(nf), sh(nf);
    std::vector<int32_t> conn(4 * nf), offs(nf), fr(nf);
    std::vector<uint8_t> types(nf, 9);
    for (size_t k = 0; k < nf; ++k) {
      for (int a = 0; a < 4; ++a) {
        int n = faces[k]->n[a];
        for (int i = 0; i < 3; ++i) {
          pts[12 * k + 3 * a + i] = mesh.X[n][i];
          disp[12 * k + 3 * a + i] = u[3 * n + i];
        }
        tt[4 * k + a] = T[n];
        conn[4 * k + a] = int32_t(4 * k + a);
      }
      offs[k] = int32_t(4 * (k + 1));
      sm[k] = smax[faces[k]->elem];
      sh[k] = shoop[faces[k]->elem];
      fr[k] = frag[faces[k]->elem];
    }
    w.f << "<Points>\n";
    w.array("Points", "Float64", 3, pts);
    w.f << "</Points>\n<Cells>\n";
    w.array("connectivity", "Int32", 1, conn);
    w.array("offsets", "Int32", 1, offs);
    w.array("types", "UInt8", 1, types);
    w.f << "</Cells>\n<PointData>\n";
    w.array("displacement_m", "Float64", 3, disp);
    w.array("temperature_K", "Float64", 1, tt);
    w.f << "</PointData>\n<CellData>\n";
    w.array("max_principal_MPa", "Float64", 1, sm);
    w.array("hoop_stress_MPa", "Float64", 1, sh);
    w.array("fragment", "Int32", 1, fr);
    w.f << "</CellData>\n";
    w.close();
  }
  // interfaces
  {
    const size_t ni = mesh.ifaces.size();
    VtuWriter w(op.dir + "/cracks_" + s + ".vtu", op.binary, 4 * ni, ni);
    std::vector<double> pts(12 * ni), disp(12 * ni), dmg(ni), dv(ni), open(ni), sg(ni), gc(ni), tn(ni), ts(ni),
        diss(ni);
    std::vector<int32_t> conn(4 * ni), offs(ni), kind(ni), plane(ni), pel(ni);
    std::vector<uint8_t> types(ni, 9);
    for (size_t f = 0; f < ni; ++f) {
      const auto& ifc = mesh.ifaces[f];
      for (int a = 0; a < 4; ++a) {
        for (int i = 0; i < 3; ++i) {
          pts[12 * f + 3 * a + i] = mesh.X[ifc.a[a]][i];
          disp[12 * f + 3 * a + i] = 0.5 * (u[3 * ifc.a[a] + i] + u[3 * ifc.b[a] + i]);
        }
        conn[4 * f + a] = int32_t(4 * f + a);
      }
      offs[f] = int32_t(4 * (f + 1));
      double d = 0, v = 0, op_ = -1e300, s1 = 0, g1 = 0, t1 = 0, t2 = 0, ds = 0;
      for (int g = 0; g < 4; ++g) {
        size_t q = f * 4 + g;
        d += 0.25 * gp_state[q].d;
        v += 0.25 * gp_state[q].dv;
        op_ = std::max(op_, double(gp_open[q]));
        s1 += 0.25 * gp_props[q].sigma_t;
        g1 += 0.25 * gp_props[q].Gc;
        t1 += 0.25 * gp_tn[q];
        t2 += 0.25 * gp_ts[q];
        ds += 0.25 * dissipated_energy(coh, gp_props[q], gp_state[q].dmax);
      }
      dmg[f] = d;
      dv[f] = v;
      open[f] = op_ * 1e6;
      sg[f] = s1 * 1e-6;
      gc[f] = g1;
      tn[f] = t1 * 1e-6;
      ts[f] = t2 * 1e-6;
      diss[f] = ds;
      kind[f] = ifc.kind;
      plane[f] = ifc.plane;
      pel[f] = ifc.pellet;
    }
    w.f << "<Points>\n";
    w.array("Points", "Float64", 3, pts);
    w.f << "</Points>\n<Cells>\n";
    w.array("connectivity", "Int32", 1, conn);
    w.array("offsets", "Int32", 1, offs);
    w.array("types", "UInt8", 1, types);
    w.f << "</Cells>\n<PointData>\n";
    w.array("displacement_m", "Float64", 3, disp);
    w.f << "</PointData>\n<CellData Scalars=\"damage\">\n";
    w.array("damage", "Float64", 1, dmg);
    w.array("viscous_damage", "Float64", 1, dv);
    w.array("opening_um", "Float64", 1, open);
    w.array("strength_MPa", "Float64", 1, sg);
    w.array("fracture_energy_Jpm2", "Float64", 1, gc);
    w.array("normal_traction_MPa", "Float64", 1, tn);
    w.array("shear_traction_MPa", "Float64", 1, ts);
    w.array("dissipated_Jpm2", "Float64", 1, diss);
    w.array("kind", "Int32", 1, kind);
    w.array("plane", "Int32", 1, plane);
    w.array("pellet", "Int32", 1, pel);
    w.f << "</CellData>\n";
    w.close();
  }
}

void Model::write_pvd() const {
  {
    std::ofstream f(op.dir + "/outputs.csv");
    f << "index,time_s\n";
    for (auto& [t, i] : vtu_list) f << i << "," << std::setprecision(10) << t << "\n";
  }
  if (!op.vtu) return;
  for (std::string kind : {"bulk", "surface", "cracks"}) {
    std::ofstream f(op.dir + "/" + kind + ".pvd");
    f << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\">\n<Collection>\n";
    for (auto& [t, i] : vtu_list)
      f << "<DataSet timestep=\"" << std::setprecision(10) << t << "\" file=\"" << kind << "_" << idx4(i)
        << ".vtu\"/>\n";
    f << "</Collection>\n</VTKFile>\n";
  }
}

void Model::write_angular(int idx) const {
  // area-weighted mean damage of radial-plane points vs angle, in bands at the
  // bottom, mid-height and top of every pellet (band = one element layer)
  const int nb = std::max(4, op.angular_bins);
  const int np = mesh.p.n_pellets;
  const double H = mesh.p.H, dz = H / mesh.p.nz;
  std::vector<double> num(size_t(np) * 3 * nb, 0.0), den(num.size(), 0.0);
  for (size_t f = 0; f < mesh.ifaces.size(); ++f) {
    if (mesh.ifaces[f].kind != IF_RADIAL) continue;
    for (int g = 0; g < 4; ++g) {
      size_t q = f * 4 + g;
      const auto& x = gp_x[q];
      int p = std::min(np - 1, int(x[2] / H));
      double zl = x[2] - p * H;
      int band = -1;
      if (zl < dz) band = 0;
      else if (std::abs(zl - 0.5 * H) < 0.5 * dz) band = 1;
      else if (zl > H - dz) band = 2;
      if (band < 0) continue;
      double th = std::atan2(x[1], x[0]);
      if (th < 0) th += 2 * M_PI;
      int b = std::min(nb - 1, int(th / (2 * M_PI) * nb));
      size_t k = (size_t(p) * 3 + band) * nb + b;
      num[k] += gp_state[q].d * gp_dA[q];
      den[k] += gp_dA[q];
    }
  }
  std::ofstream f(op.dir + "/angular_" + idx4(idx) + ".csv");
  f << "theta_deg";
  const char* bn[3] = {"bottom", "mid", "top"};
  for (int p = 0; p < np; ++p)
    for (int b = 0; b < 3; ++b) f << ",p" << p << "_" << bn[b] << "_damage";
  f << "\n";
  for (int b = 0; b < nb; ++b) {
    f << (b + 0.5) * 360.0 / nb;
    for (int p = 0; p < np; ++p)
      for (int bd = 0; bd < 3; ++bd) {
        size_t k = (size_t(p) * 3 + bd) * nb + b;
        f << "," << (den[k] > 0 ? num[k] / den[k] : 0.0);
      }
    f << "\n";
  }
}

void Model::write_interface_profile(int idx) const {
  if (mesh.p.n_pellets < 2) return;
  std::ofstream f(op.dir + "/interface_" + idx4(idx) + ".csv");
  f << "x_mm,y_mm,z_mm,r_mm,theta_deg,area_mm2,pressure_MPa,shear_MPa,gap_um,sliding,damage\n";
  for (size_t ff = 0; ff < mesh.ifaces.size(); ++ff) {
    if (mesh.ifaces[ff].kind != IF_PELLET) continue;
    for (int g = 0; g < 4; ++g) {
      size_t q = ff * 4 + g;
      const auto& x = gp_x[q];
      double th = std::atan2(x[1], x[0]) * 180 / M_PI;
      if (th < 0) th += 360;
      f << x[0] * 1e3 << "," << x[1] * 1e3 << "," << x[2] * 1e3 << "," << std::hypot(x[0], x[1]) * 1e3 << "," << th
        << "," << gp_dA[q] * 1e6 << "," << std::max(0.0, -double(gp_tn[q])) * 1e-6 << "," << gp_ts[q] * 1e-6 << ","
        << gp_open[q] * 1e6 << "," << gp_slip[q] << "," << gp_state[q].d << "\n";
    }
  }
}

void Model::write_fragments(const std::string& path, std::vector<double>* stats) const {
  std::vector<int> lab, erid;
  int nf = label_fragments(lab, erid);
  std::vector<double> vol(nf, 0.0), cx(nf, 0.0), cy(nf, 0.0), cz(nf, 0.0);
  std::vector<int> pel(nf, 0), nreg(nf, 0);
  std::vector<char> seen(lab.size(), 0);
  for (size_t e = 0; e < mesh.nelem(); ++e) {
    int fr = lab[erid[e]];
    double c[3] = {0, 0, 0};
    for (int a = 0; a < 8; ++a)
      for (int i = 0; i < 3; ++i) c[i] += mesh.X[mesh.hex[e][a]][i] / 8;
    vol[fr] += elem_vol[e];
    cx[fr] += c[0] * elem_vol[e];
    cy[fr] += c[1] * elem_vol[e];
    cz[fr] += c[2] * elem_vol[e];
    pel[fr] = mesh.elem_pellet[e];
    if (!seen[erid[e]]) {
      seen[erid[e]] = 1;
      nreg[fr]++;
    }
  }
  std::ofstream f(path);
  f << "fragment,pellet,n_regions,volume_mm3,d_eq_mm,cx_mm,cy_mm,cz_mm\n";
  for (int i = 0; i < nf; ++i)
    f << i << "," << pel[i] << "," << nreg[i] << "," << vol[i] * 1e9 << "," << std::cbrt(6 * vol[i] / M_PI) * 1e3
      << "," << cx[i] / vol[i] * 1e3 << "," << cy[i] / vol[i] * 1e3 << "," << cz[i] / vol[i] * 1e3 << "\n";
  if (stats) {
    stats->assign(size_t(mesh.p.n_pellets) * 3, 0.0);  // n, d_eq volume-weighted, sum vol
    for (int i = 0; i < nf; ++i) {
      double d = std::cbrt(6 * vol[i] / M_PI);
      (*stats)[3 * pel[i]] += 1;
      (*stats)[3 * pel[i] + 1] += d * vol[i];
      (*stats)[3 * pel[i] + 2] += vol[i];
    }
  }
}

void Model::write_summary(bool complete, const std::string& msg) const {
  std::vector<double> fs;
  write_fragments(op.dir + "/fragments.csv", &fs);
  // crack area per plane
  std::map<std::pair<int, int>, double> plane_area;
  for (size_t f = 0; f < mesh.ifaces.size(); ++f) {
    const auto& ifc = mesh.ifaces[f];
    if (ifc.kind == IF_PELLET && !mesh.p.cohesive_pellet_interface) continue;
    for (int g = 0; g < 4; ++g)
      if (gp_state[f * 4 + g].d >= coh.d_crack) plane_area[{ifc.kind, ifc.plane}] += gp_dA[f * 4 + g];
  }
  const double V = M_PI * mesh.p.R * mesh.p.R * mesh.p.H;
  std::ofstream f(op.dir + "/summary.json");
  f << "{\n \"complete\": " << (complete ? "true" : "false") << ",\n \"message\": " << jstr(msg)
    << ",\n \"steps\": " << step << ",\n \"time_s\": " << jnum(time)
    << ",\n \"wall_s\": " << jnum(last.count("wall_s") ? last.at("wall_s") : 0.0)
    << ",\n \"build_id\": " << jstr(build_identifier());
  for (auto& [k, v] : last) f << ",\n \"final_" << k << "\": " << jnum(v);
  f << ",\n \"onset_time_s\": " << jnum(first_crack_time < 0 ? NAN : first_crack_time)
    << ",\n \"onset_dT_K\": " << jnum(first_crack_dT < 0 ? NAN : first_crack_dT)
    << ",\n \"onset_linear_power_Wpm\": " << jnum(first_crack_power < 0 ? NAN : first_crack_power)
    << ",\n \"crack_area_final_mm2\": [";
  for (int p = 0; p < mesh.p.n_pellets; ++p) f << (p ? ", " : "") << jnum(met.pellet_area[p] * 1e6);
  f << "],\n \"crack_area_by_kind_mm2\": {";
  for (int k = 0; k < IF_NKINDS; ++k) f << (k ? ", " : "") << jstr(iface_kind_name(k)) << ": " << jnum(met.area[k] * 1e6);
  f << "},\n \"crack_area_by_plane_mm2\": [";
  bool first = true;
  for (auto& [key, a] : plane_area) {
    f << (first ? "" : ", ") << "{\"kind\": " << jstr(iface_kind_name(key.first)) << ", \"plane\": " << key.second
      << ", \"area_mm2\": " << jnum(a * 1e6) << "}";
    first = false;
  }
  f << "],\n \"fragments_final\": [";
  for (int p = 0; p < mesh.p.n_pellets; ++p) {
    double n = fs[3 * p], dvw = fs[3 * p + 2] > 0 ? fs[3 * p + 1] / fs[3 * p + 2] : NAN;
    double A = met.pellet_area[p];
    f << (p ? ", " : "") << "{\"pellet\": " << p << ", \"n\": " << int(n)
      << ", \"deq_volweighted_mm\": " << jnum(dvw * 1e3) << ", \"Leq_mm\": " << jnum(A > 0 ? 3 * V / A * 1e3 : NAN)
      << ", \"specific_area\": " << jnum(A * mesh.p.R / V) << "}";
  }
  f << "]\n}\n";
}

}  // namespace pcz
