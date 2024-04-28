// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include <CLI/CLI.hpp>

#include "two_scale_capillarity.hpp"

// Main function to run the program
//
int main(int argc, char* argv[]) {
  // Mesh parameters
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {0.0, 0.0};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {0.75, 0.75};
  std::size_t min_level = 8;
  std::size_t max_level = 8;

  // Simulation parameters
  double Tf  = 1e-4;
  double cfl = 0.5;

  bool apply_relaxation = true;

  CLI::App app{"Finite volume example for the two-phase static bubble "
                "using multiresolution"};
  app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
  app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
  app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
  CLI11_PARSE(app, argc, argv);

  // Output parameters
  std::size_t nfiles = 100;

  // Create the instance of the class to perform the simulation
  auto StaticBubble_Sim = StaticBubble(min_corner, max_corner, min_level, max_level,
                                       Tf, cfl, nfiles, apply_relaxation);

  StaticBubble_Sim.run();

  return 0;
}
