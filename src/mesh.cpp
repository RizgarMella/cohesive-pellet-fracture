#include "mesh.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "config.hpp"

namespace pcz {

const int HEX_FACES[6][4] = {
    {0, 3, 2, 1},  // zeta = -1 (bottom)
    {4, 5, 6, 7},  // zeta = +1 (top)
    {0, 1, 5, 4},  // eta  = -1 (theta-)
    {1, 2, 6, 5},  // xi   = +1 (outer)
    {2, 3, 7, 6},  // eta  = +1 (theta+)
    {3, 0, 4, 7},  // xi   = -1 (inner)
};

const char* iface_kind_name(int k) {
  switch (k) {
    case IF_RADIAL: return "radial";
    case IF_TRANSVERSE: return "transverse";
    case IF_CIRCUMFERENTIAL: return "circumferential";
    case IF_PELLET: return "pellet_interface";
  }
  return "?";
}

void MeshParams::read(const Config& c) {
  R = c.get_double("geometry.radius", R);
  H = c.get_double("geometry.height", H);
  n_pellets = c.get_int("geometry.n_pellets", n_pellets);
  nr = c.get_int("mesh.nr", nr);
  ntheta = c.get_int("mesh.ntheta", ntheta);
  nz = c.get_int("mesh.nz", nz);
  radial_grading = c.get_double("mesh.radial_grading", radial_grading);
  std::string nrp = c.get_string("cracks.radial_planes", std::to_string(n_radial_planes));
  n_radial_planes = (nrp == "all") ? -1 : std::stoi(nrp);
  theta_offset_deg = c.get_list("cracks.theta_offset_deg", {0.0});
  circ_radii = c.get_list("cracks.circumferential_radii", {});
  transverse_z = c.get_list("cracks.transverse_z", {});
  if (c.has("cracks.n_slabs") && !c.has("cracks.transverse_z")) {
    int ns = c.get_int("cracks.n_slabs", 1);
    transverse_z.clear();
    for (int i = 1; i < ns; ++i) transverse_z.push_back(double(i) / ns);
  }
  cohesive_pellet_interface = c.get_string("interface.type", "contact") == "cohesive";
}

namespace {
struct ArrHash {
  size_t operator()(const std::array<int, 4>& a) const {
    size_t h = 1469598103934665603ull;
    for (int v : a) h = (h ^ size_t(v)) * 1099511628211ull;
    return h;
  }
};
}  // namespace

void Mesh::build(const MeshParams& mp) {
  p = mp;
  const int nr = p.nr, nth = p.ntheta, nzp = p.nz, np = p.n_pellets;
  const int nzt = nzp * np;
  if (nr < 1 || nth < 3 || nzp < 1 || np < 1) throw std::runtime_error("mesh: bad resolution");
  if (p.n_radial_planes == 1) throw std::runtime_error("cracks.radial_planes must be 0, >=2 or 'all'");
  int nsec = 1, spsec = nth;  // sectors, elements per sector
  if (p.n_radial_planes == -1) {
    nsec = nth;
    spsec = 1;
  } else if (p.n_radial_planes >= 2) {
    if (nth % p.n_radial_planes) throw std::runtime_error("mesh.ntheta must be a multiple of cracks.radial_planes");
    nsec = p.n_radial_planes;
    spsec = nth / nsec;
  }

  // 1D coordinates
  r_nodes.resize(nr + 1);
  for (int i = 0; i <= nr; ++i) {
    double s = double(i) / nr;
    r_nodes[i] = p.R * (1.0 - std::pow(1.0 - s, p.radial_grading));
  }
  z_nodes.resize(nzt + 1);
  for (int k = 0; k <= nzt; ++k) z_nodes[k] = p.H * double(k) / nzp;

  // snapped crack positions
  std::vector<int> ring_cut;  // node ring indices where circumferential cracks lie
  for (double f : p.circ_radii) {
    double rr = f * p.R;
    int best = 1;
    for (int i = 1; i < nr; ++i)
      if (std::abs(r_nodes[i] - rr) < std::abs(r_nodes[best] - rr)) best = i;
    if (nr > 1) ring_cut.push_back(best);
  }
  std::sort(ring_cut.begin(), ring_cut.end());
  ring_cut.erase(std::unique(ring_cut.begin(), ring_cut.end()), ring_cut.end());
  std::vector<int> z_cut;  // local z node index within a pellet
  for (double f : p.transverse_z) {
    int k = int(std::lround(f * nzp));
    if (k > 0 && k < nzp) z_cut.push_back(k);
  }
  std::sort(z_cut.begin(), z_cut.end());
  z_cut.erase(std::unique(z_cut.begin(), z_cut.end()), z_cut.end());
  std::vector<int> th_off(np, 0);
  for (int q = 0; q < np; ++q) {
    double deg = p.theta_offset_deg.empty() ? 0.0 : p.theta_offset_deg[std::min<size_t>(q, p.theta_offset_deg.size() - 1)];
    th_off[q] = int(std::lround(deg / 360.0 * nth));
  }

  // original (continuous) nodes
  const int per_level = 1 + nr * nth;
  n_orig_nodes = per_level * (nzt + 1);
  auto onode = [&](int i, int j, int k) { return k * per_level + (i == 0 ? 0 : 1 + (i - 1) * nth + ((j % nth) + nth) % nth); };
  std::vector<std::array<double, 3>> Xo(n_orig_nodes);
  for (int k = 0; k <= nzt; ++k) {
    Xo[onode(0, 0, k)] = {0.0, 0.0, z_nodes[k]};
    for (int i = 1; i <= nr; ++i)
      for (int j = 0; j < nth; ++j) {
        double th = 2.0 * M_PI * j / nth;
        Xo[onode(i, j, k)] = {r_nodes[i] * std::cos(th), r_nodes[i] * std::sin(th), z_nodes[k]};
      }
  }

  // elements in original numbering + regions
  const int ne = nr * nth * nzt;
  std::vector<std::array<int, 8>> hex_o(ne);
  elem_region.assign(ne, {0, 0, 0, 0});
  elem_pellet.assign(ne, 0);
  std::vector<std::array<int, 3>> eijk(ne);
  int nslab = int(z_cut.size()) + 1, nring = int(ring_cut.size()) + 1;
  for (int k = 0; k < nzt; ++k)
    for (int j = 0; j < nth; ++j)
      for (int i = 0; i < nr; ++i) {
        int e = (k * nth + j) * nr + i;
        eijk[e] = {i, j, k};
        hex_o[e] = {onode(i, j, k),     onode(i + 1, j, k),     onode(i + 1, j + 1, k),     onode(i, j + 1, k),
                    onode(i, j, k + 1), onode(i + 1, j, k + 1), onode(i + 1, j + 1, k + 1), onode(i, j + 1, k + 1)};
        int pel = k / nzp, kl = k % nzp;
        int slab = 0;
        for (int zc : z_cut)
          if (kl >= zc) ++slab;
        int ring = 0;
        for (int rc : ring_cut)
          if (i >= rc) ++ring;
        int sec = (((j - th_off[pel]) % nth) + nth) % nth / spsec;
        elem_region[e] = {pel, slab, sec, ring};
        elem_pellet[e] = pel;
      }
  auto region_id = [&](int e) {
    auto& r = elem_region[e];
    return ((r[0] * nslab + r[1]) * nsec + r[2]) * nring + r[3];
  };

  // node -> elements
  std::vector<std::vector<int>> n2e(n_orig_nodes);
  for (int e = 0; e < ne; ++e)
    for (int a = 0; a < 8; ++a) {
      auto& v = n2e[hex_o[e][a]];
      if (v.empty() || v.back() != e) v.push_back(e);
    }
  // duplicate nodes per region
  std::vector<std::vector<std::pair<int, int>>> copies(n_orig_nodes);  // (region, new id)
  X.clear();
  for (int n = 0; n < n_orig_nodes; ++n) {
    for (int e : n2e[n]) {
      int rg = region_id(e);
      bool found = false;
      for (auto& pr : copies[n])
        if (pr.first == rg) found = true;
      if (!found) {
        copies[n].push_back({rg, int(X.size())});
        X.push_back(Xo[n]);
      }
    }
  }
  auto copy_of = [&](int n, int rg) {
    for (auto& pr : copies[n])
      if (pr.first == rg) return pr.second;
    throw std::runtime_error("mesh: missing node copy");
  };
  hex.resize(ne);
  for (int e = 0; e < ne; ++e) {
    int rg = region_id(e);
    for (int a = 0; a < 8; ++a) hex[e][a] = copy_of(hex_o[e][a], rg);
  }

  // faces: interfaces and boundaries
  std::unordered_map<std::array<int, 4>, std::pair<int, int>, ArrHash> facemap;
  facemap.reserve(size_t(ne) * 3);
  ifaces.clear();
  outer.clear();
  bottom.clear();
  top.clear();
  for (int e = 0; e < ne; ++e) {
    for (int f = 0; f < 6; ++f) {
      std::array<int, 4> q;
      for (int a = 0; a < 4; ++a) q[a] = hex_o[e][HEX_FACES[f][a]];
      std::array<int, 4> key = q;
      std::sort(key.begin(), key.end());
      // collapsed hexes at the axis give triangular faces (one repeated node,
      // kept) and line faces on the axis itself (two repeated nodes, skipped)
      int distinct = 1 + (key[1] != key[0]) + (key[2] != key[1]) + (key[3] != key[2]);
      if (distinct < 3) continue;
      auto [i, j, k] = eijk[e];
      if (f == 3 && i == nr - 1) {
        BFace bf;
        for (int a = 0; a < 4; ++a) bf.n[a] = hex[e][HEX_FACES[f][a]];
        bf.elem = e;
        outer.push_back(bf);
        continue;
      }
      if (f == 0 && k == 0) {
        BFace bf;
        for (int a = 0; a < 4; ++a) bf.n[a] = hex[e][HEX_FACES[f][a]];
        bf.elem = e;
        bottom.push_back(bf);
        continue;
      }
      if (f == 1 && k == nzt - 1) {
        BFace bf;
        for (int a = 0; a < 4; ++a) bf.n[a] = hex[e][HEX_FACES[f][a]];
        bf.elem = e;
        top.push_back(bf);
        continue;
      }
      auto it = facemap.find(key);
      if (it == facemap.end()) {
        facemap.emplace(key, std::make_pair(e, f));
        continue;
      }
      int eA = it->second.first, fA = it->second.second;
      facemap.erase(it);
      int rA = region_id(eA), rB = region_id(e);
      if (rA == rB) continue;
      Iface ifc;
      std::array<int, 4> qa;
      for (int a = 0; a < 4; ++a) qa[a] = hex_o[eA][HEX_FACES[fA][a]];
      for (int a = 0; a < 4; ++a) {
        ifc.a[a] = copy_of(qa[a], rA);
        ifc.b[a] = copy_of(qa[a], rB);
      }
      auto &RA = elem_region[eA], &RB = elem_region[e];
      auto [iA, jA, kA] = eijk[eA];
      ifc.pellet = RA[0];
      if (RA[0] != RB[0]) {
        ifc.kind = IF_PELLET;
        ifc.plane = std::max(kA, k);
      } else if (RA[2] != RB[2]) {
        ifc.kind = IF_RADIAL;
        // theta node index of the plane: the face nodes share j (or j+1)
        ifc.plane = (fA == 2) ? jA : (jA + 1) % nth;
      } else if (RA[1] != RB[1]) {
        ifc.kind = IF_TRANSVERSE;
        ifc.plane = std::max(kA, k);
      } else {
        ifc.kind = IF_CIRCUMFERENTIAL;
        ifc.plane = std::max(iA, i);
      }
      ifaces.push_back(ifc);
    }
  }
  build_graph();
}

void Mesh::build_graph() {
  const int nn = int(X.size());
  std::vector<std::vector<int>> nb(nn);
  auto link = [&](const int* ids, int m) {
    for (int a = 0; a < m; ++a)
      for (int b = 0; b < m; ++b) nb[ids[a]].push_back(ids[b]);
  };
  for (auto& h : hex) link(h.data(), 8);
  for (auto& f : ifaces) {
    int ids[8] = {f.a[0], f.a[1], f.a[2], f.a[3], f.b[0], f.b[1], f.b[2], f.b[3]};
    link(ids, 8);
  }
  adj_ptr.assign(nn + 1, 0);
  adj.clear();
  for (int n = 0; n < nn; ++n) {
    auto& v = nb[n];
    v.push_back(n);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    adj_ptr[n + 1] = adj_ptr[n] + int(v.size());
    adj.insert(adj.end(), v.begin(), v.end());
  }
}

std::string Mesh::summary() const {
  std::ostringstream o;
  int cnt[IF_NKINDS] = {0, 0, 0, 0};
  for (auto& f : ifaces) cnt[f.kind]++;
  o << "mesh: " << nelem() << " hex, " << nnode() << " nodes (" << n_orig_nodes << " before crack-plane duplication), "
    << ifaces.size() << " interface elements [radial " << cnt[0] << ", transverse " << cnt[1] << ", circumferential "
    << cnt[2] << ", pellet-pellet " << cnt[3] << "]";
  return o.str();
}

}  // namespace pcz
