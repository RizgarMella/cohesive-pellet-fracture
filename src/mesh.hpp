// Structured 3D hexahedral mesh of a stack of cylindrical pellets with
// zero-thickness interface elements inserted on user-selected potential
// crack planes (radial, transverse, circumferential) and on the
// pellet-pellet contact plane.
//
// Construction: a polar (r, theta, z) hex mesh is generated (the innermost
// ring uses collapsed hexes = wedges at the axis). Every element gets a
// "region" tuple (pellet, slab, sector, ring) from the crack-plane layout.
// Each geometric node is then duplicated once per region touching it, and
// every element face shared by two different regions becomes a 4+4 node
// interface element. This handles junctions of crossing crack planes
// (e.g. radial planes meeting at the axis) without special cases.
#pragma once
#include <array>
#include <string>
#include <vector>

namespace pcz {

class Config;

enum IfaceKind { IF_RADIAL = 0, IF_TRANSVERSE = 1, IF_CIRCUMFERENTIAL = 2, IF_PELLET = 3, IF_NKINDS = 4 };
const char* iface_kind_name(int k);

struct MeshParams {
  double R = 4.67e-3;   // pellet radius [m]
  double H = 14.0e-3;   // pellet height [m]
  int n_pellets = 1;
  int nr = 12;          // radial elements
  int ntheta = 32;      // circumferential elements
  int nz = 24;          // axial elements per pellet
  double radial_grading = 1.0;  // >1 clusters elements toward the outer surface
  // crack layout
  int n_radial_planes = 8;               // 0 = none, -1 = every element boundary
  std::vector<double> theta_offset_deg;  // per pellet rotation of radial planes
  std::vector<double> circ_radii;        // fractions of R (snapped to node rings)
  std::vector<double> transverse_z;      // fractions of H within each pellet (snapped)
  bool cohesive_pellet_interface = false;  // bonded (cohesive) instead of contact

  void read(const Config& c);
};

struct Iface {
  std::array<int, 4> a;  // nodes on side A (normal points from A to B)
  std::array<int, 4> b;  // nodes on side B
  int kind;              // IfaceKind
  int plane;             // index identifying the plane (theta, ring or z index)
  int pellet;            // pellet of side A
};

struct BFace {
  std::array<int, 4> n;  // nodes, ordered so the normal is outward
  int elem;
};

struct Mesh {
  MeshParams p;
  std::vector<std::array<double, 3>> X;  // node coordinates (after duplication)
  std::vector<std::array<int, 8>> hex;   // hex connectivity
  std::vector<int> elem_pellet;
  std::vector<std::array<int, 4>> elem_region;  // pellet, slab, sector, ring
  std::vector<Iface> ifaces;
  std::vector<BFace> outer, bottom, top;  // boundary faces
  std::vector<double> r_nodes, z_nodes;   // 1D node coordinates
  int n_orig_nodes = 0;
  // node adjacency graph (CSR, includes self) used for the sparse patterns
  std::vector<int> adj_ptr, adj;

  size_t nnode() const { return X.size(); }
  size_t nelem() const { return hex.size(); }
  void build(const MeshParams& mp);
  void build_graph();
  std::string summary() const;
};

// local face node lists (outward normal) for the standard hex ordering
extern const int HEX_FACES[6][4];

}  // namespace pcz
