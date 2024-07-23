// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

#include "flux_6eqs.hpp"

//#define HLLC_FLUX
#define HLLC_BR_FLUX
//#define RUSANOV_FLUX

// Specify the use of this namespace where we just store the indices
// and some parameters related to the equations of state
using namespace EquationData;

// This is the class for the simulation of a two-scale model
//
template<std::size_t dim>
class Relaxation {
public:
  using Config = samurai::MRConfig<dim, 2>;

  Relaxation() = default; // Default constructor. This will do nothing
                          // and basically will never be used

  Relaxation(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
             const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
             std::size_t min_level, std::size_t max_level,
             double Tf_, double cfl_, std::size_t nfiles_ = 100,
             bool do_pres_relax = true, bool do_pres_reinit = true,
             bool do_energy_update_phase_1 = true, bool do_preserve_energy = false); // Class constrcutor with the arguments related
                                                                                     // to the grid, to the physics and to the relaxation.
                                                                                     // Maybe in the future,
                                                                                     // we could think to add parameters related to EOS

  void run(); // Function which actually executes the temporal loop

  template<class... Variables>
  void save(const fs::path& path,
            const std::string& filename,
            const std::string& suffix,
            const Variables&... fields); // Routine to save the results

private:
  /*--- Now we declare some relevant variables ---*/
  const samurai::Box<double, dim> box;

  samurai::MRMesh<Config> mesh; // Variable to store the mesh

  using Field        = samurai::Field<decltype(mesh), double, EquationData::NVARS, false>;
  using Field_Scalar = samurai::Field<decltype(mesh), double, 1, false>;
  using Field_Vect   = samurai::Field<decltype(mesh), double, dim, false>;

  double Tf;  // Final time of the simulation
  double cfl; // Courant number of the simulation so as to compute the time step

  std::size_t nfiles; // Number of files desired for output

  bool apply_pressure_relax;        // Set whether to apply or not the pressure relaxation
  bool apply_pressure_reinit;       // Set whether to apply or not the reinitialization step for the pressure
  bool start_energy_update_phase_1; // Start the energy update from phase 1 or 2
  bool preserve_energy;             // Set how to update the total energy during the pressure relaxation

  Field conserved_variables; // The variable which stores the conserved variables,
                             // namely the varialbes for which we solve a PDE system

  const SG_EOS<> EOS_phase1; // Equation of state of phase 1
  const SG_EOS<> EOS_phase2; // Equation of state of phase 2

  #ifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> numerical_flux_cons; // variable to compute the numerical flux for the conservative part
                                                     // (this is necessary to call 'make_flux')

    samurai::NonConservativeFlux<Field> numerical_flux_non_cons; // variable to compute the numerical flux for the non-conservative part
                                                                 // (this is necessary to call 'make_flux')
  #elifdef HLLC_FLUX
    samurai::HLLCFlux<Field> numerical_flux; // variable to compute the numerical flux
                                             // (this is necessary to call 'make_flux')
  #elifdef HLLC_BR_FLUX
    samurai::HLLCFlux_Conservative<Field> numerical_flux_cons; // variable to compute the numerical flux for the conservative part
                                                               // (this is necessary to call 'make_flux')

    samurai::NonConservativeFlux<Field> numerical_flux_non_cons; // variable to compute the numerical flux for the non-conservative part
                                                                 // (this is necessary to call 'make_flux')
  #endif

  /*--- Now we declare a bunch of fields which depend from the state,
        but it is useful to have it for the output ---*/
  Field_Scalar rho,
               p,
               rho1,
               p1,
               c1,
               rho2,
               p2,
               c2,
               c,
               alpha2,
               Y2;

  Field_Vect vel;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void init_variables(); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  void update_auxiliary_fields(); // Routine to update auxilairy fields for output and time step update

  double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue

  void update_pressure_before_relaxation(); // Update pressure fields before relaxation

  void apply_instantaneous_pressure_relaxation(); // Apply an instantaneous pressure relaxation

  void apply_instantaneous_pressure_relaxation_Saurel();
};

// Implement class constructor
//
#if defined RUSANOV_FLUX || defined HLLC_BR_FLUX
  template<std::size_t dim>
  Relaxation<dim>::Relaxation(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                              const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                              std::size_t min_level, std::size_t max_level,
                              double Tf_, double cfl_, std::size_t nfiles_,
                              bool do_pres_relax, bool do_pres_reinit,
                              bool do_energy_update_phase_1, bool do_preserve_energy):
    box(min_corner, max_corner), mesh(box, min_level, max_level, {false}),
    Tf(Tf_), cfl(cfl_), nfiles(nfiles_),
    apply_pressure_relax(do_pres_relax), apply_pressure_reinit(do_pres_reinit),
    start_energy_update_phase_1(do_energy_update_phase_1), preserve_energy(do_preserve_energy),
    EOS_phase1(EquationData::gamma_1, EquationData::pi_infty_1, EquationData::q_infty_1),
    EOS_phase2(EquationData::gamma_2, EquationData::pi_infty_2, EquationData::q_infty_2),
    numerical_flux_cons(EOS_phase1, EOS_phase2),
    numerical_flux_non_cons(EOS_phase1, EOS_phase2) {
      std::cout << "Initializing variables" << std::endl;
      std::cout << std::endl;
      init_variables();
    }
#elifdef HLLC_FLUX
  template<std::size_t dim>
  Relaxation<dim>::Relaxation(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                              const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                              std::size_t min_level, std::size_t max_level,
                              double Tf_, double cfl_, std::size_t nfiles_,
                              bool do_pres_relax, bool do_pres_reinit,
                              bool do_energy_update_phase_1, bool do_preserve_energy):
    box(min_corner, max_corner), mesh(box, min_level, max_level, {false}),
    Tf(Tf_), cfl(cfl_), nfiles(nfiles_),
    apply_pressure_relax(do_pres_relax), apply_pressure_reinit(do_pres_reinit),
    start_energy_update_phase_1(do_energy_update_phase_1), preserve_energy(do_preserve_energy),
    EOS_phase1(EquationData::gamma_1, EquationData::pi_infty_1, EquationData::q_infty_1),
    EOS_phase2(EquationData::gamma_2, EquationData::pi_infty_2, EquationData::q_infty_2),
    numerical_flux(EOS_phase1, EOS_phase2) {
      std::cout << "Initializing variables" << std::endl;
      std::cout << std::endl;
      init_variables();
    }
#endif

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void Relaxation<dim>::init_variables() {
  /*--- Create conserved and auxiliary fields ---*/
  conserved_variables = samurai::make_field<double, EquationData::NVARS>("conserved", mesh);

  rho    = samurai::make_field<double, 1>("rho", mesh);
  p      = samurai::make_field<double, 1>("p", mesh);

  rho1   = samurai::make_field<double, 1>("rho1", mesh);
  p1     = samurai::make_field<double, 1>("p1", mesh);
  c1     = samurai::make_field<double, 1>("c1", mesh);

  rho2   = samurai::make_field<double, 1>("rho2", mesh);
  p2     = samurai::make_field<double, 1>("p2", mesh);
  c2     = samurai::make_field<double, 1>("c2", mesh);

  c      = samurai::make_field<double, 1>("c", mesh);

  vel    = samurai::make_field<double, dim>("vel", mesh);

  alpha2 = samurai::make_field<double, 1>("alpha2", mesh);
  Y2     = samurai::make_field<double, 1>("Y2", mesh);

  /*--- Set the initial state ---*/
  const double xd = 0.5;

  // Initialize the fields with a loop over all cells
  const double alpha1L = 1.0 - 1e-2;

  const double velL    = -2.0;

  const double p1L     = 1e5;
  const double rho1L   = 1150.0;

  const double alpha2L = 1.0 - alpha1L;

  const double p2L     = 1e5;
  const double rho2L   = 0.63;

  const double alpha1R = alpha1L;

  const double velR    = 2.0;

  const double p1R     = 1e5;
  const double rho1R   = 1150.0;

  const double alpha2R = 1.0 - alpha1R;

  const double p2R     = 1e5;
  const double rho2R   = 0.63;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];

                           /*--- Left state (primitive variables) ---*/
                           if(x <= xd) {
                             conserved_variables[cell][ALPHA1_INDEX] = alpha1L;

                             vel[cell]  = velL;

                             p1[cell]   = p1L;
                             rho1[cell] = rho1L;

                             p2[cell]   = p2L;
                             rho2[cell] = rho2L;
                           }
                           /*--- Right state (primitive variables) ---*/
                           else {
                             conserved_variables[cell][ALPHA1_INDEX] = alpha1R;

                             vel[cell]  = velR;

                             p1[cell]   = p1R;
                             rho1[cell] = rho1R;

                             p2[cell]   = p2R;
                             rho2[cell] = rho2R;
                           }

                           /*--- Complete the conserved variables (and some auxiliary fields for the sake of completeness) ---*/
                           conserved_variables[cell][ALPHA1_RHO1_INDEX] = conserved_variables[cell][ALPHA1_INDEX]*rho1[cell];

                           conserved_variables[cell][ALPHA2_RHO2_INDEX] = (1.0 - conserved_variables[cell][ALPHA1_INDEX])*rho2[cell];

                           rho[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX];
                           conserved_variables[cell][RHO_U_INDEX] = rho[cell]*vel[cell];

                           const auto e1 = EOS_phase1.e_value(rho1[cell], p1[cell]);
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*(e1 + 0.5*vel[cell]*vel[cell]);

                           const auto e2 = EOS_phase2.e_value(rho2[cell], p2[cell]);
                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*(e2 + 0.5*vel[cell]*vel[cell]);

                           c1[cell] = EOS_phase1.c_value(rho1[cell], p1[cell]);

                           c2[cell] = EOS_phase2.c_value(rho2[cell], p2[cell]);

                           p[cell] = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                   + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*p2[cell];

                           c[cell] = std::sqrt((conserved_variables[cell][ALPHA1_RHO1_INDEX]/rho[cell])*c1[cell]*c1[cell] +
                                               (1.0 - conserved_variables[cell][ALPHA1_RHO1_INDEX]/rho[cell])*c2[cell]*c2[cell]);

                           alpha2[cell] = 1.0 - conserved_variables[cell][ALPHA1_INDEX];
                           Y2[cell]     = conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho[cell];
                         });


  const xt::xtensor_fixed<int, xt::xshape<1>> left{-1};
  const xt::xtensor_fixed<int, xt::xshape<1>> right{1};
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables, alpha1L, alpha1L*rho1L, alpha2L*rho2L, (alpha1L*rho1L + alpha2L*rho2L)*velL,
                                                                        alpha1L*rho1L*(EOS_phase1.e_value(rho1L, p1L) + 0.5*velL*velL),
                                                                        alpha2L*rho2L*(EOS_phase2.e_value(rho2L, p2L) + 0.5*velL*velL))->on(left);
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables, alpha1R, alpha1R*rho1R, alpha2R*rho2R, (alpha1R*rho1R + alpha2R*rho2R)*velR,
                                                                        alpha1R*rho1R*(EOS_phase1.e_value(rho1R, p1R) + 0.5*velR*velR),
                                                                        alpha2R*rho2R*(EOS_phase2.e_value(rho2R, p2R) + 0.5*velR*velR))->on(right);
}

// Update pressure fields before relaxation
//
template<std::size_t dim>
void Relaxation<dim>::update_pressure_before_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           auto e1 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                               vanishing volume fraction ---*/

                           auto e2 = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                               vanishing volume fraction ---*/
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             const auto vel_d = conserved_variables[cell][RHO_U_INDEX + d]/
                                                (conserved_variables[cell][ALPHA1_RHO1_INDEX] + conserved_variables[cell][ALPHA2_RHO2_INDEX]);

                             e1 -= 0.5*vel_d*vel_d;
                             e2 -= 0.5*vel_d*vel_d;
                           }
                           p1[cell] = EOS_phase1.pres_value(conserved_variables[cell][ALPHA1_RHO1_INDEX]/conserved_variables[cell][ALPHA1_INDEX], e1);
                           /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           p2[cell] = EOS_phase2.pres_value(conserved_variables[cell][ALPHA2_RHO2_INDEX]/(1.0 - conserved_variables[cell][ALPHA1_INDEX]), e2);
                           /*--- TODO: Add treatment for vanishing volume fraction ---*/
                         });
}

// Apply the instantaneous relaxation for the pressure
//
template<std::size_t dim>
void Relaxation<dim>::apply_instantaneous_pressure_relaxation_Saurel() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           /*--- Save mixture total energy for later update ---*/
                           const auto rhoE_0 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]
                                             + conserved_variables[cell][ALPHA2_RHO2_E2_INDEX];

                           /*--- Compute the pressure equilibirum with the polynomial method (Saurel) ---*/
                           const auto b = (conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2*(EquationData::pi_infty_2 - p1[cell]) +
                                           (1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1*(EquationData::pi_infty_1 - p2[cell]))/
                                          (conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2 + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1);
                           const auto c = -(p1[cell]*conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2*EquationData::pi_infty_2 +
                                            p2[cell]*(1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1*EquationData::pi_infty_1)/
                                           (conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2 + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1);

                           auto p_star = 0.5*(-b + std::sqrt(b*b - 4.0*c));

                           /*--- Update the volume fraction using the computed pressure ---*/
                           conserved_variables[cell][ALPHA1_INDEX] *= (p1[cell] + EquationData::gamma_1*EquationData::pi_infty_1 + p_star*(EquationData::gamma_1 - 1.0))/
                                                                      (p_star + EquationData::gamma_1*EquationData::pi_infty_1 + p_star*(EquationData::gamma_1 - 1.0));

                           /*--- Apply the pressure reinitialization if desired ---*/
                           if(apply_pressure_reinit) {
                             const auto rho_0 = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                              + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                             auto rhoe_0 = rhoE_0;
                             for(std::size_t d = 0; d < EquationData::dim; ++d) {
                               // Compute the equilibrium velocity
                               // (use the mixture law, so as to avoid divsion by zero, even though it is the same for each phase)
                               const auto vel_d = conserved_variables[cell][RHO_U_INDEX + d]/rho_0;
                               rhoe_0 -= 0.5*rho_0*vel_d*vel_d;
                             }
                             p_star = (rhoe_0 - (conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_1*EquationData::pi_infty_1/(EquationData::gamma_1 - 1.0) +
                                                 (1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_2*EquationData::pi_infty_2/(EquationData::gamma_2 - 1.0)))/
                                      (conserved_variables[cell][ALPHA1_INDEX]/(EquationData::gamma_1 - 1.0) +
                                       (1.0 - conserved_variables[cell][ALPHA1_INDEX])/(EquationData::gamma_2 - 1.0));
                           }

                           /*--- Update the total energy starting from phase 1 ---*/
                           if(start_energy_update_phase_1) {
                             // Update the total energy of phase 1
                             auto E1 = EOS_phase1.e_value(conserved_variables[cell][ALPHA1_RHO1_INDEX]/conserved_variables[cell][ALPHA1_INDEX], p_star);
                             /*--- TODO: Add treatment for vanishing volume fraction ---*/

                             for(std::size_t d = 0; d < EquationData::dim; ++d) {
                               const auto vel_d = conserved_variables[cell][RHO_U_INDEX + d]/
                                                  (conserved_variables[cell][ALPHA1_RHO1_INDEX] + conserved_variables[cell][ALPHA2_RHO2_INDEX]);

                               E1 += 0.5*vel_d*vel_d;
                             }

                             conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*E1;

                             if(preserve_energy) {
                               // Update the total energy of phase 2 imposing total energy conservation
                               conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = rhoE_0 - conserved_variables[cell][ALPHA1_RHO1_E1_INDEX];
                             }
                             else {
                               // Update the total energy of phase 2
                               auto E2 = EOS_phase2.e_value(conserved_variables[cell][ALPHA2_RHO2_INDEX]/(1.0 - conserved_variables[cell][ALPHA1_INDEX]), p_star);
                               /*--- TODO: Add treatment for vanishing volume fraction ---*/

                               for(std::size_t d = 0; d < EquationData::dim; ++d) {
                                 const auto vel_d = conserved_variables[cell][RHO_U_INDEX + d]/
                                                    (conserved_variables[cell][ALPHA1_RHO1_INDEX] + conserved_variables[cell][ALPHA2_RHO2_INDEX]);

                                 E2 += 0.5*vel_d*vel_d;
                               }

                               conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*E2;
                             }
                           }
                           /*--- Update the total energy starting from phase 2 ---*/
                           else {
                             // Update the total energy of phase 2
                             auto E2 = EOS_phase2.e_value(conserved_variables[cell][ALPHA2_RHO2_INDEX]/(1.0 - conserved_variables[cell][ALPHA1_INDEX]), p_star);
                             /*--- TODO: Add treatment for vanishing volume fraction ---*/

                             for(std::size_t d = 0; d < EquationData::dim; ++d) {
                               const auto vel_d = conserved_variables[cell][RHO_U_INDEX + d]/
                                                  (conserved_variables[cell][ALPHA1_RHO1_INDEX] + conserved_variables[cell][ALPHA2_RHO2_INDEX]);

                               E2 += 0.5*vel_d*vel_d;
                             }

                             conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*E2;

                             if(preserve_energy) {
                               // Update the total energy of phase 1 imposing total energy conservation
                               conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = rhoE_0 - conserved_variables[cell][ALPHA2_RHO2_E2_INDEX];
                             }
                             else {
                               // Update the total energy of phase 1
                               auto E1 = EOS_phase1.e_value(conserved_variables[cell][ALPHA1_RHO1_INDEX]/conserved_variables[cell][ALPHA1_INDEX], p_star);
                               /*--- TODO: Add treatment for vanishing volume fraction ---*/

                               for(std::size_t d = 0; d < EquationData::dim; ++d) {
                                 const auto vel_d = conserved_variables[cell][RHO_U_INDEX + d]/
                                                    (conserved_variables[cell][ALPHA1_RHO1_INDEX] + conserved_variables[cell][ALPHA2_RHO2_INDEX]);

                                 E1 += 0.5*vel_d*vel_d;
                               }

                               conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*E1;
                             }
                           }
                         });
}




template<std::size_t dim>
void Relaxation<dim>::apply_instantaneous_pressure_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                            /*--- Save some quantities which remain constant during relaxation ---*/
                            const auto rhoE_0 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]
                                             + conserved_variables[cell][ALPHA2_RHO2_E2_INDEX];
                            
                            const auto arho1_0 = conserved_variables[cell][ALPHA1_RHO1_INDEX];
                            const auto arho2_0 = conserved_variables[cell][ALPHA2_RHO2_INDEX];

                            const auto rho_0 = arho1_0 + arho2_0;
                            
                            auto vel_squared = 0.0;
                            for(std::size_t d = 0; d < EquationData::dim; ++d) {
                              const auto vel_d = conserved_variables[cell][RHO_U_INDEX + d]/rho_0;
                              vel_squared += vel_d*vel_d;
                            }

                            const auto e_0 = rhoE_0 / rho_0 - 0.5*vel_squared;
                            
                            const auto Y1_0 = arho1_0 / rho_0;
                            const auto Y2_0 = 1.0 - Y1_0;

                            const auto gammaM1_1 = EquationData::gamma_1 - 1.0;
                            const auto gammaM1_2 = EquationData::gamma_2 - 1.0;

                            const auto gampinf_1 = EquationData::gamma_1 * EquationData::pi_infty_1;
                            const auto gampinf_2 = EquationData::gamma_2 * EquationData::pi_infty_2;

                            

                            /*---  ----*/
                            auto alpha_1 = conserved_variables[cell][ALPHA1_INDEX];
                            auto alpha_2 = 1.0 - alpha_1;
                            
                            auto rho_1 = arho1_0 / alpha_1;
                            auto rho_2 = arho2_0 / alpha_2;

                            auto p_1 = p1[cell];
                            auto p_2 = p2[cell];

                            auto e_1 = EOS_phase1.e_value(rho_1, p_1);
                            auto e_2 = EOS_phase2.e_value(rho_2, p_2);

                            /*--- for the moment p_I = p_1 ---*/
                            auto &p_I = p_1;
                            const auto Laplace_cst_1 = (p_1 + EquationData::pi_infty_1)/std::pow(rho_1,EquationData::gamma_1);
                            const auto Laplace_cst_2 = (p_2 + EquationData::pi_infty_2)/std::pow(rho_2,EquationData::gamma_2);

                            auto dalpha = 0.0;

                            /*
                            auto dp1de1 = 0.0;
                            auto dp2de2  = 0.0;

                            auto dp1drho1 = 0.0;
                            auto dp2drho2  = 0.0;
                            */

                            int nite = 0;
                            auto alpha_max = 1.0;
                            auto alpha_min = 0.0;

                            while (nite < 100 && 2.0*(alpha_max-alpha_min)/(alpha_max+alpha_min)>1e-8) {
                              /*
                              dp1de1 = gammaM1_1 * rho_1;
                              dp2de2 = gammaM1_2 * rho_2;
                              dp1drho1 = gammaM1_1 * e_1;
                              dp1drho1 = gammaM1_2 * e_2;
                              
                              dalpha = (p_1 - p_2) /
                                       (  arho1_0*dp1drho1/(alpha_1*alpha_1) + p_I*dp1de1/arho1_0
                                        + arho2_0*dp2drho2/(alpha_2*alpha_2) + p_I*dp2de2/arho2_0 );
                              */

                              p_1>p_2 ? alpha_min=alpha_1 : alpha_max=alpha_1;
                              
                              dalpha = (p_1 - p_2) /
                                       std::abs( (p_1 + gammaM1_1*p_I + gampinf_1)/alpha_1
                                        +(p_2 + gammaM1_2*p_I + gampinf_2)/alpha_2 );

                              dalpha = std::min(dalpha, 0.9*(alpha_max-alpha_1));
                              dalpha = std::max(dalpha, 0.9*(alpha_min-alpha_1));
                              alpha_1 += dalpha;
                              alpha_2 -= dalpha;

                              rho_1 = arho1_0 / alpha_1;
                              rho_2 = arho2_0 / alpha_2;

                              
                              p_1 = std::pow(rho_1,EquationData::gamma_1)*Laplace_cst_1 - EquationData::pi_infty_1;
                              e_1 = EOS_phase1.e_value(rho_1, p_1);
                              e_2 = (e_0 - Y1_0*e_1)/Y2_0;
                              p_2 = EOS_phase2.pres_value(rho_2, e_2);
                              /*
                              p_2 = std::pow(rho_2,EquationData::gamma_2)*Laplace_cst_2 - EquationData::pi_infty_2;
                              e_2 = EOS_phase2.e_value(rho_2, p_2);
                              e_1 = (e_0 - Y2_0*e_2)/Y1_0;
                              p_1 = EOS_phase1.pres_value(rho_1, e_1);
                              */
                              
                              nite += 1;
                            }
                            
                            /*--- Update the conserved variables : Alpha, AlphaRhoE_1, AlphaRhoE_2 ---*/
                            conserved_variables[cell][ALPHA1_INDEX] = alpha_1;
                            conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = arho1_0*(e_1 + 0.5*vel_squared);
                            conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = rhoE_0 - conserved_variables[cell][ALPHA1_RHO1_E1_INDEX];
                          });
}





// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double Relaxation<dim>::get_max_lambda() const {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           res = std::max(std::max(std::abs(vel[cell]) + c1[cell],
                                                   std::abs(vel[cell]) + c2[cell]),
                                          res);
                         });

  return res;
}

// Update auxiliary fields after solution of the system
//
template<std::size_t dim>
void Relaxation<dim>::update_auxiliary_fields() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           rho[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           /*--- Phase 1 ---*/
                           rho1[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]/conserved_variables[cell][ALPHA1_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                          vanishing volume fraction ---*/
                           vel[cell] = conserved_variables[cell][RHO_U_INDEX]/rho[cell];

                           auto e1 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                               vanishing volume fraction ---*/
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             e1 -= 0.5*vel[cell]*vel[cell];
                           }
                           p1[cell] = EOS_phase1.pres_value(rho1[cell], e1);
                           c1[cell] = EOS_phase1.c_value(rho1[cell], p1[cell]);

                           /*--- Phase 2 ---*/
                           rho2[cell] = conserved_variables[cell][ALPHA2_RHO2_INDEX]/(1.0 - conserved_variables[cell][ALPHA1_INDEX]); /*--- TODO: Add treatment for
                                                                                                                                                  vanishing volume fraction ---*/

                           auto e2 = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                               vanishing volume fraction ---*/
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             e2 -= 0.5*vel[cell]*vel[cell];
                           }
                           p2[cell] = EOS_phase2.pres_value(rho2[cell], e2);
                           c2[cell] = EOS_phase2.c_value(rho2[cell], p2[cell]);

                           alpha2[cell] = 1.0 - conserved_variables[cell][ALPHA1_INDEX];
                           Y2[cell]     = conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho[cell];

                           /*--- Remaining mixture variables ---*/
                           p[cell] = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                   + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*p2[cell];

                           c[cell] = std::sqrt((conserved_variables[cell][ALPHA1_RHO1_INDEX]/rho[cell])*c1[cell]*c1[cell] +
                                               (1.0 - conserved_variables[cell][ALPHA1_RHO1_INDEX]/rho[cell])*c2[cell]*c2[cell]);
                         });
}

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void Relaxation<dim>::save(const fs::path& path,
                           const std::string& filename,
                           const std::string& suffix,
                           const Variables&... fields) {
  auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

  if(!fs::exists(path)) {
    fs::create_directory(path);
  }

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           level_[cell] = cell.level;
                         });

  samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, fields..., level_);
}

// Implement the function that effectively performs the temporal loop
//
template<std::size_t dim>
void Relaxation<dim>::run() {
  /*--- Default output arguemnts ---*/
  fs::path path        = fs::current_path();
  std::string filename;
  #ifdef RUSANOV_FLUX
    filename = "Relaxation_Rusanov_6eqs_total_energy";
  #elifdef HLLC_FLUX
    filename = "Relaxation_HLLC_6eqs_total_energy";
  #elifdef HLLC_BR_FLUX
    filename = "Relaxation_HLLC_BR_6eqs_total_energy";
  #endif
  const double dt_save = Tf / static_cast<double>(nfiles);

  /*--- Auxiliary variables to save updated fields ---*/
  auto conserved_variables_tmp   = samurai::make_field<double, EquationData::NVARS>("conserved_tmp", mesh);
  auto conserved_variables_tmp_2 = samurai::make_field<double, EquationData::NVARS>("conserved_tmp_2", mesh);
  auto conserved_variables_np1   = samurai::make_field<double, EquationData::NVARS>("conserved_np1", mesh);

  /*--- Create the flux variables ---*/
  #ifdef RUSANOV_FLUX
    auto Rusanov_flux         = numerical_flux_cons.make_flux();
    auto NonConservative_flux = numerical_flux_non_cons.make_flux();
  #elifdef HLLC_FLUX
    auto HLLC_flux = numerical_flux.make_flux();
  #elifdef HLLC_BR_FLUX
    auto HLLC_Conservative_flux = numerical_flux_cons.make_flux();
    auto NonConservative_flux   = numerical_flux_non_cons.make_flux();
  #endif

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, conserved_variables, rho, p, vel, c, rho1, p1, c1, rho2, p2, c2, alpha2, Y2);

  /*--- Save mesh size ---*/
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  const double dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  double t          = 0.0;
  double dt         = std::min(Tf - t, cfl*dx/get_max_lambda());
  while(t != Tf) {
    t += dt;

    std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

    // Apply the numerical scheme (first stage)
    samurai::update_ghost_mr(conserved_variables);
    samurai::update_bc(conserved_variables);
    #ifdef RUSANOV_FLUX
      auto Cons_Flux          = Rusanov_flux(conserved_variables);
      auto NonCons_Flux       = NonConservative_flux(conserved_variables);
      conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
    #elifdef HLLC_FLUX
      auto Total_Flux         = HLLC_flux(conserved_variables);
      conserved_variables_tmp = conserved_variables - dt*Total_Flux;
    #elifdef HLLC_BR_FLUX
      auto Cons_Flux          = HLLC_Conservative_flux(conserved_variables);
      auto NonCons_Flux       = NonConservative_flux(conserved_variables);
      conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
    #endif

    std::swap(conserved_variables.array(), conserved_variables_tmp.array());

    if(apply_pressure_relax) {
      // Apply the relaxation for the pressure (first stage)
      update_pressure_before_relaxation();
      apply_instantaneous_pressure_relaxation();
    }
    /*
    // Apply the numerical scheme (second stage)
    samurai::update_ghost_mr(conserved_variables);
    samurai::update_bc(conserved_variables);
    #ifdef RUSANOV_FLUX
      Cons_Flux                 = Rusanov_flux(conserved_variables);
      NonCons_Flux              = NonConservative_flux(conserved_variables);
      conserved_variables_tmp_2 = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
    #elifdef HLLC_FLUX
      Total_Flux                = HLLC_flux(conserved_variables);
      conserved_variables_tmp_2 = conserved_variables - dt*Total_Flux;
    #elifdef HLLC_BR_FLUX
      Cons_Flux                 = HLLC_Conservative_flux(conserved_variables);
      NonCons_Flux              = NonConservative_flux(conserved_variables);
      conserved_variables_tmp_2 = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
    #endif
    conserved_variables_np1 = 0.5*(conserved_variables_tmp + conserved_variables_tmp_2);
    std::swap(conserved_variables.array(), conserved_variables_np1.array());

    if(apply_pressure_relax) {
      // Apply the relaxation for the pressure (second stage)
      update_pressure_before_relaxation();
      apply_instantaneous_pressure_relaxation();
    }
    */
    // Compute updated time step
    update_auxiliary_fields();
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix, conserved_variables, rho, p, vel, c, rho1, p1, c1, rho2, p2, c2, alpha2, Y2);
    }
  }
}
