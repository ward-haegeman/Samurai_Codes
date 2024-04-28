// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include "two_scale_capillarity.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  constexpr std::size_t dim = 2;

  // Mesh parameters
  xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0.0, 0.0};
  xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {4.0, 2.0};
  std::size_t min_level = 8;
  std::size_t max_level = 8;

  // Simulation parameters
  double Tf  = 2.0;
  double cfl = 0.4;

  bool apply_relaxation = true;

  bool mass_transfer = true;

  // Output parameters
  std::size_t nfiles = 20;

  // Create the instance of the class to perform the simulation
  auto TwoScaleCapillarity_Sim = TwoScaleCapillarity(min_corner, max_corner, min_level, max_level,
                                                     Tf, cfl, nfiles, apply_relaxation, mass_transfer);

  TwoScaleCapillarity_Sim.run();

  return 0;
}
