// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
//#define FULL_BN
#define TOT_ENERGY_6EQS
//#define INT_ENERGY_6EQS

#ifdef FULL_BN
  #include "relaxation.hpp"
#elifdef TOT_ENERGY_6EQS
  #include "relaxation_6eqs.hpp"
#elifdef INT_ENERGY_6EQS
  #include "relaxation_6eqs_int_energy.hpp"
#endif

// Main function to run the program
//
int main(int argc, char* argv[]) {
  /*--- Mesh parameters ---*/
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> min_corner = {0.0};
  xt::xtensor_fixed<double, xt::xshape<EquationData::dim>> max_corner = {1.0};
  std::size_t min_level = 9;
  std::size_t max_level = 9;

  /*--- Simulation parameters ---*/
  double Tf  = 3.2e-3;
  double cfl = 0.5;

  /*--- Output parameters ---*/
  std::size_t nfiles = 20;

  /*--- Perform the simulation ---*/
  #ifdef FULL_BN
    bool apply_velocity_relax  = true;
    bool apply_pressure_relax  = true;
    bool apply_pressure_reinit = false;
    bool energy_update_phase_1 = true;
    bool preserve_energy       = false;

    // Create the instance of the class to perform the simulation
    auto Relaxation_Sim = Relaxation(min_corner, max_corner, min_level, max_level,
                                     Tf, cfl, nfiles,
                                     apply_velocity_relax, apply_pressure_relax,
                                     apply_pressure_reinit, energy_update_phase_1,
                                     preserve_energy);

    Relaxation_Sim.run();
  #endif
  #ifdef TOT_ENERGY_6EQS
    bool apply_pressure_relax  = true;
    bool apply_pressure_reinit = false;
    bool energy_update_phase_1 = true;
    bool preserve_energy       = false;

    // Create the instance of the class to perform the simulation
    auto Relaxation_Sim = Relaxation(min_corner, max_corner, min_level, max_level,
                                     Tf, cfl, nfiles,
                                     apply_pressure_relax, apply_pressure_reinit,
                                     energy_update_phase_1, preserve_energy);

    Relaxation_Sim.run();
  #endif
  #ifdef INT_ENERGY_6EQS
    bool apply_pressure_relax  = true;
    bool apply_pressure_reinit = false;
    bool energy_update_phase_1 = true;
    bool preserve_energy       = false;

    // Create the instance of the class to perform the simulation
    auto Relaxation_Rusanov_Sim = Relaxation(min_corner, max_corner, min_level, max_level,
                                             Tf, cfl, nfiles,
                                             apply_pressure_relax, apply_pressure_reinit,
                                             energy_update_phase_1, preserve_energy);

    Relaxation_Rusanov_Sim.run();
  #endif

  return 0;
}
