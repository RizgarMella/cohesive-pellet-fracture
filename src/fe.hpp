// Finite-element kernels: trilinear hexahedron (2x2x2 Gauss) and bilinear
// quadrilateral surfaces (2x2 Gauss) used for boundaries and interfaces.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace pcz {

constexpr double GP1 = 0.57735026918962576451;  // 1/sqrt(3)
constexpr int HEX_SIGN[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                                {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
constexpr int QUAD_SIGN[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};

inline void hex_shape(double xi, double eta, double zeta, double N[8], double dN[8][3]) {
  for (int a = 0; a < 8; ++a) {
    double sx = HEX_SIGN[a][0], sy = HEX_SIGN[a][1], sz = HEX_SIGN[a][2];
    double fx = 1 + sx * xi, fy = 1 + sy * eta, fz = 1 + sz * zeta;
    N[a] = 0.125 * fx * fy * fz;
    dN[a][0] = 0.125 * sx * fy * fz;
    dN[a][1] = 0.125 * sy * fx * fz;
    dN[a][2] = 0.125 * sz * fx * fy;
  }
}

// Gauss point g (0..7) of the hex: shape functions, global gradients and dV.
// Returns false for a non-positive Jacobian.
inline bool hex_gp(const double X[8][3], int g, double N[8], double B[8][3], double& dV) {
  double xi = HEX_SIGN[g][0] * GP1, eta = HEX_SIGN[g][1] * GP1, zeta = HEX_SIGN[g][2] * GP1;
  double dN[8][3];
  hex_shape(xi, eta, zeta, N, dN);
  double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  for (int a = 0; a < 8; ++a)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) J[i][j] += dN[a][j] * X[a][i];  // J_ij = dx_i/dxi_j
  double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
               J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
  if (det <= 0) {
    for (int a = 0; a < 8; ++a) B[a][0] = B[a][1] = B[a][2] = 0.0;
    dV = 0.0;
    return false;
  }
  double Ji[3][3];
  double id = 1.0 / det;
  Ji[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * id;
  Ji[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * id;
  Ji[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * id;
  Ji[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * id;
  Ji[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * id;
  Ji[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * id;
  Ji[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * id;
  Ji[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * id;
  Ji[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * id;
  // dN/dx_k = sum_j dN/dxi_j * dxi_j/dx_k = dN_j * Ji[j][k]
  for (int a = 0; a < 8; ++a)
    for (int k = 0; k < 3; ++k) B[a][k] = dN[a][0] * Ji[0][k] + dN[a][1] * Ji[1][k] + dN[a][2] * Ji[2][k];
  dV = det;  // Gauss weights are 1 for the 2-point rule
  return true;
}

// Quad Gauss point g (0..3): shape functions, area element and orthonormal
// frame (rows: normal, tangent1, tangent2). Normal follows the node order.
inline void quad_gp(const double X[4][3], int g, double N[4], double& dA, double R[3][3], double x[3]) {
  double xi = QUAD_SIGN[g][0] * GP1, eta = QUAD_SIGN[g][1] * GP1;
  double dxi[3] = {0, 0, 0}, deta[3] = {0, 0, 0};
  x[0] = x[1] = x[2] = 0;
  for (int a = 0; a < 4; ++a) {
    double sx = QUAD_SIGN[a][0], sy = QUAD_SIGN[a][1];
    N[a] = 0.25 * (1 + sx * xi) * (1 + sy * eta);
    double dNx = 0.25 * sx * (1 + sy * eta), dNy = 0.25 * sy * (1 + sx * xi);
    for (int i = 0; i < 3; ++i) {
      dxi[i] += dNx * X[a][i];
      deta[i] += dNy * X[a][i];
      x[i] += N[a] * X[a][i];
    }
  }
  double n[3] = {dxi[1] * deta[2] - dxi[2] * deta[1], dxi[2] * deta[0] - dxi[0] * deta[2],
                 dxi[0] * deta[1] - dxi[1] * deta[0]};
  dA = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  double t1n = std::sqrt(dxi[0] * dxi[0] + dxi[1] * dxi[1] + dxi[2] * dxi[2]);
  for (int i = 0; i < 3; ++i) {
    R[0][i] = n[i] / dA;
    R[1][i] = dxi[i] / t1n;
  }
  // t2 = n x t1
  R[2][0] = R[0][1] * R[1][2] - R[0][2] * R[1][1];
  R[2][1] = R[0][2] * R[1][0] - R[0][0] * R[1][2];
  R[2][2] = R[0][0] * R[1][1] - R[0][1] * R[1][0];
}

// eigenvalues of a symmetric 3x3 tensor (s = xx, yy, zz, xy, yz, xz), descending
inline void sym_eig3(const double s[6], double ev[3]) {
  double a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5];
  double p1 = d * d + e * e + f * f;
  double q = (a + b + c) / 3.0;
  if (p1 < 1e-30 * (a * a + b * b + c * c + 1e-300)) {
    ev[0] = a;
    ev[1] = b;
    ev[2] = c;
  } else {
    double p2 = (a - q) * (a - q) + (b - q) * (b - q) + (c - q) * (c - q) + 2 * p1;
    double p = std::sqrt(p2 / 6.0);
    double B[6] = {(a - q) / p, (b - q) / p, (c - q) / p, d / p, e / p, f / p};
    double detB = B[0] * (B[1] * B[2] - B[4] * B[4]) - B[3] * (B[3] * B[2] - B[4] * B[5]) +
                  B[5] * (B[3] * B[4] - B[1] * B[5]);
    double r = std::min(1.0, std::max(-1.0, detB / 2.0));
    double phi = std::acos(r) / 3.0;
    ev[0] = q + 2 * p * std::cos(phi);
    ev[2] = q + 2 * p * std::cos(phi + 2.0 * M_PI / 3.0);
    ev[1] = 3 * q - ev[0] - ev[2];
  }
  // sort descending
  if (ev[0] < ev[1]) std::swap(ev[0], ev[1]);
  if (ev[1] < ev[2]) std::swap(ev[1], ev[2]);
  if (ev[0] < ev[1]) std::swap(ev[0], ev[1]);
}

}  // namespace pcz
