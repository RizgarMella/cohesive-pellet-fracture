// pelletcz - cohesive-zone thermal fracture of UO2 pellets (single pellets and
// pellet stacks) with Weibull-distributed fracture properties.
//
//   pelletcz input.ini [section.key=value ...]          one simulation (OpenMP threads)
//   pelletcz --ensemble cases.txt [--threads N]          many simulations; with MPI the
//   mpirun -n R pelletcz --ensemble cases.txt            lines are distributed over ranks
//   pelletcz --dry-run input.ini                         mesh + property statistics only
//
// cases.txt: one case per line, "input.ini key=value key=value ...", '#' comments.
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef PELLETCZ_HAVE_MPI
#include <mpi.h>
#endif

#include "config.hpp"
#include "model.hpp"

using namespace pcz;

static int run_case(const std::vector<std::string>& args, bool dry) {
  Config cfg;
  cfg.load_file(args.at(0));
  for (size_t i = 1; i < args.size(); ++i) {
    auto eq = args[i].find('=');
    if (eq == std::string::npos) throw std::runtime_error("override must be section.key=value: " + args[i]);
    cfg.set(trim(args[i].substr(0, eq)), trim(args[i].substr(eq + 1)));
  }
  Model m(cfg);
  if (dry) {
    m.setup();
    std::cout << "dry run: " << m.mesh.nnode() * 3 << " displacement dofs, " << m.ngp << " interface points\n";
  } else {
    m.run();
  }
  for (auto& k : cfg.unused_keys()) std::cerr << "warning: unused input key '" << k << "'\n";
  return 0;
}

static void usage() {
  std::cout << "usage: pelletcz input.ini [section.key=value ...]\n"
               "       pelletcz --ensemble cases.txt [--threads N]   (MPI: mpirun -n R pelletcz --ensemble ...)\n"
               "       pelletcz --dry-run input.ini [overrides]\n";
}

int main(int argc, char** argv) {
  int rank = 0, size = 1;
#ifdef PELLETCZ_HAVE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
  int status = 0;
  try {
    std::vector<std::string> a(argv + 1, argv + argc);
    std::string ensemble;
    bool dry = false;
    std::vector<std::string> rest;
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] == "--ensemble" && i + 1 < a.size())
        ensemble = a[++i];
      else if (a[i] == "--threads" && i + 1 < a.size()) {
#ifdef _OPENMP
        omp_set_num_threads(std::stoi(a[++i]));
#else
        ++i;
#endif
      } else if (a[i] == "--dry-run")
        dry = true;
      else if (a[i] == "-h" || a[i] == "--help") {
        usage();
        return 0;
      } else
        rest.push_back(a[i]);
    }
#ifdef _OPENMP
    if (rank == 0) std::cout << "pelletcz: " << omp_get_max_threads() << " OpenMP threads per case";
#else
    if (rank == 0) std::cout << "pelletcz: serial build";
#endif
    if (rank == 0) std::cout << ", " << size << " MPI rank(s), build " << build_identifier() << "\n";
    if (!ensemble.empty()) {
      std::ifstream f(ensemble);
      if (!f) throw std::runtime_error("cannot open ensemble file " + ensemble);
      std::vector<std::vector<std::string>> cases;
      std::string line;
      while (std::getline(f, line)) {
        auto c = line.find('#');
        if (c != std::string::npos) line = line.substr(0, c);
        std::istringstream ss(line);
        std::vector<std::string> toks;
        std::string t;
        while (ss >> t) toks.push_back(t);
        if (!toks.empty()) cases.push_back(toks);
      }
      int failed = 0;
      for (size_t i = 0; i < cases.size(); ++i) {
        if (int(i % size) != rank) continue;
        std::cout << "[rank " << rank << "] case " << i + 1 << "/" << cases.size() << ": " << cases[i][0] << "\n";
        try {
          run_case(cases[i], dry);
        } catch (std::exception& e) {
          std::cerr << "[rank " << rank << "] case " << i + 1 << " failed: " << e.what() << "\n";
          ++failed;
        }
      }
      status = failed ? 2 : 0;
    } else {
      if (rest.empty()) {
        usage();
        status = 1;
      } else {
        status = run_case(rest, dry);
      }
    }
  } catch (std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    status = 1;
  }
#ifdef PELLETCZ_HAVE_MPI
  MPI_Finalize();
#endif
  return status;
}
