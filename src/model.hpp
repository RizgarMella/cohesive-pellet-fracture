// The coupled thermo-mechanical cohesive-zone pellet model.
#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "config.hpp"
#include "interface_law.hpp"
#include "materials.hpp"
#include "mesh.hpp"
#include "random_field.hpp"
#include "sparse.hpp"
#include "amg.hpp"

namespace pcz {

// piecewise-linear history, constant outside its range
struct Table {
  std::vector<double> t, v;
  double operator()(double x) const;
  bool empty() const { return t.empty(); }
  void set_constant(double c) {
    t = {0.0};
    v = {c};
  }
};

struct TimeParams {
  double t_end = 200.0;
  double dt0 = 1.0, dt_min = 1e-9, dt_max = 1e7;
  double grow = 1.5, cut = 0.25;
  double max_dpower_frac = 0.05;   // max change of power per step (fraction of max power)
  double max_damage_increment = 0.5;
  double softening_dt_factor = 2.0;  // dt <= factor * viscosity while cracks are growing
  double initiation_target = 1.02;   // approach crack initiation in steps (failure index)
  int max_steps = 1000000;
  std::vector<double> stops;       // times the stepping must hit (history breakpoints, outputs)
};

struct SolverParams {
  int newton_max = 40;
  double newton_tol = 5e-4;        // ||R|| / ||F_ref||
  double newton_dtol = 1e-6;       // ||du||_inf / max|u| (secondary)
  double lin_rtol = 1e-3;          // inexact Newton: PCG to 1e-3 of the current residual
  int lin_maxit = 20000;
  double stabilization = 1e-9;     // matrix-only diagonal shift (fraction of mean diagonal)
  int thermal_max = 30;
  double thermal_tol = 1e-3;       // K
  int coupling_iters = 1;          // staggered thermal -> mechanical passes per step
  double thermal_skip_dt = 1e-2;   // s: steps shorter than this keep the temperature field
  bool line_search = true;
  bool amg = true;                 // AMG preconditioner (false: block Jacobi)
  int amg_rebuild_iters = 40;      // rebuild when PCG needs more than this AND twice the fresh-hierarchy count
};

struct ThermalBC {
  double h_conv = 5000.0;   // W/m2/K
  Table T_coolant;          // K
  bool fixed_surface = false;  // Dirichlet T = coolant instead of convection
  double emissivity = 0.8;
  double roughness = 1.0e-6;   // m, added to crack opening in k_g/(d_g + r)
  double h_bonded = 1.0e7;     // W/m2/K across intact interfaces (~1000x the element conductance k/h)
  double h_contact = 5.0e4;    // W/m2/K solid contact conductance at the pellet interface
  double T_initial = 500.0;
};

struct MechBC {
  std::string bottom = "roller";  // roller | platen
  double platen_friction = 0.0;
  double platen_stiffness = 1e16;
  std::string top = "free";       // free | pressure | displacement
  Table top_pressure;             // Pa
  Table top_displacement;         // m (uz, negative = pressing down)
  bool pin_rigid_body = true;
};

struct OutputParams {
  std::string dir = "out";
  std::vector<double> times;   // VTU output times
  int every = 0;               // also every N accepted steps (0 = off)
  double on_crack_growth = 0;  // also when crack area grew by this [m2] since last VTU
  bool vtu = true;
  bool binary = true;
  int max_vtu = 400;
  int angular_bins = 72;
  std::vector<std::string> labels;
};

struct Diagnostics {  // per step
  int newton_iters = 0, lin_iters = 0, thermal_iters = 0;
  double res = 0;
};

class Model {
public:
  explicit Model(const Config& cfg);
  void run();

  // ---- setup (model.cpp)
  void setup();
  void assign_fracture_properties();
  // ---- thermal (thermal.cpp)
  bool solve_thermal(double t, double dt, int& iters);
  // ---- mechanics (mechanics.cpp)
  void assemble_bulk_stiffness();
  void compute_eigen_forces();
  bool solve_mechanics(double t, double dt, Diagnostics& dg);
  void interface_forces(double dt, std::vector<double>* F, bool assemble);
  void element_stress(int e, double sig[6]) const;
  // ---- outputs (output.cpp)
  void write_manifest() const;
  void open_history();
  void write_history_row(const Diagnostics& dg);
  void write_vtu(int idx);
  void write_pvd() const;
  void write_summary(bool complete, const std::string& msg) const;
  void write_angular(int idx) const;
  void write_interface_profile(int idx) const;
  void write_fragments(const std::string& path, std::vector<double>* stats) const;

  // global quantities
  double power_density(double t) const;  // W/m3
  double linear_power(double t) const;   // W/m

  const Config& cfg;
  Mesh mesh;
  UO2 mat;
  CohesiveParams coh;
  ContactParams con;
  WeibullSpec strength, toughness;
  double corr_length = 0.5e-3, tough_corr = 1.0, ref_area = 0.0;
  int points_per_length = 4;
  uint64_t seed = 1;
  std::vector<std::vector<double>> flaws;
  double dish_inner_radius = 0.0, dish_depth = 0.3e-3, chamfer_width = 0.0, chamfer_depth = 0.1e-3;
  TimeParams tp;
  SolverParams sp;
  ThermalBC tbc;
  MechBC mbc;
  OutputParams op;
  Table power;  // W/m (linear heat rate)

  // interface integration points (4 per interface element)
  size_t ngp = 0;
  std::vector<double> gp_dA;
  std::vector<std::array<double, 9>> gp_R;
  std::vector<std::array<double, 3>> gp_x;
  std::vector<GPProps> gp_props;
  std::vector<GPState> gp_state, gp_trial;
  std::vector<float> gp_open, gp_tn, gp_ts, gp_slip;  // last evaluated opening / normal traction / slip flag
  int n_gc_clamped = 0;
  double failure_index = 0.0;   // max delta_e/delta_0 over intact cohesive points (last evaluation)
  int n_softening = 0;          // cohesive points on the softening branch (last evaluation)

  // bottom platen (nodal contact) data
  std::vector<int> platen_node;
  std::vector<double> platen_area;
  std::vector<GPState> platen_state, platen_trial;

  // fields
  std::vector<double> T, T_old, u, u_old;
  std::vector<double> elem_vol;
  std::vector<double> gas_swell, gas_swell_trial;  // per element volumetric strain
  double fima = 0.0, fima_trial = 0.0;
  double time = 0.0;
  std::vector<double> mask_u, mask_T;  // Dirichlet masks
  std::vector<int> top_nodes, bottom_nodes, pin_dofs;
  int centre_node = 0;
  std::vector<int> surf_nodes;      // outer surface, mid-height of pellet 0
  std::vector<double> Rres;         // residual at the end of the last mechanical solve

  // matrices
  BlockCSR Kb;      // bulk stiffness (constant)
  BlockCSR Kmat;    // working matrix
  BlockCSR Kth;     // thermal
  AMG amg_u, amg_T;
  int last_pcg_u = 1 << 30, last_pcg_T = 1 << 30, amg_builds = 0, amg_builds_T = 0, base_pcg_u = 1, base_pcg_T = 1;
  long pcg_T_total = 0;
  std::vector<std::array<int, 64>> hex_blk, if_blk;  // block positions for scatter
  std::vector<double> F_eig, F_ext;
  double Fref = 1.0, Fref_dev = 1.0;

  // bookkeeping
  int step = 0, n_vtu = 0;
  std::vector<std::pair<double, int>> vtu_list;
  double last_vtu_area = 0.0;
  double first_crack_time = -1, first_crack_dT = -1, first_crack_power = -1;
  double wall_start = 0.0, tm_thermal = 0, tm_mech = 0, tm_metrics = 0;
  std::string build_id;
  std::FILE* hist = nullptr;
  std::map<std::string, double> last;  // last history row values (for summary)

  // metrics
  struct Metrics {
    double area[IF_NKINDS] = {0, 0, 0, 0};          // cracked area per kind [m2]
    double damaged_area = 0, dissipated = 0;          // [m2], [J]
    double Tc = 0, Ts = 0, Tmax = 0;
    double max_principal = 0;
    // pellet interface
    double if_force = 0, if_area = 0, if_pmax = 0, if_slip_frac = 0, if_shear_force = 0;
    double bottom_reaction = 0, top_reaction = 0, pin_reaction = 0;
    std::vector<double> pellet_area;  // cracked area per pellet [m2]
    int n_fragments = 0;
  } met;
  // fragment labelling: regions joined by intact ligaments
  int label_fragments(std::vector<int>& region_label, std::vector<int>& elem_region_id) const;
  void compute_metrics();
};

std::string build_identifier();

}  // namespace pcz
