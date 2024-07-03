// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include "BN_solver.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  // Mesh parameters
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {0.0, 0.0};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {1.0, 0.5};
  std::size_t min_level = 7;
  std::size_t max_level = 7;

  // Simulation parameters
  double Tf  = 0.007;
  double cfl = 0.45;

  // Output parameters
  std::size_t nfiles = 20;

  // Create the instance of the class to perform the simulation
  auto BN_Solver_Sim = BN_Solver(min_corner, max_corner, min_level, max_level,
                                 Tf, cfl, nfiles);

  BN_Solver_Sim.run();

  return 0;
}
