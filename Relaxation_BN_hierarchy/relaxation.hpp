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

#include "flux.hpp"

#define SULICIU_RELAXATION
//#define RUSANOV_FLUX

#define RELAX_POLYNOM

// Specify the use of this namespace where we just store the indices
// and some parameters related to the equations of state
using namespace EquationData;

// This is the class for the simulation of a BN model
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
             bool do_vel_relax = true, bool do_pres_relax = true,
             bool do_pres_reinit = true, bool do_energy_update_phase_1 = true,
             bool do_preserve_energy = false); // Class constrcutor with the arguments related
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

  bool apply_velocity_relax;        // Set whether to apply or not the velocity relaxation
  bool apply_pressure_relax;        // Set whether to apply or not the pressure relaxation
  bool apply_pressure_reinit;       // Set whether to apply or not the reinitialization step for the pressure
  bool start_energy_update_phase_1; // Start the energy update from phase 1 or 2
  bool preserve_energy;             // Set how to update the total energy of phase for the pressure relaxation

  Field conserved_variables; // The variable which stores the conserved variables,
                             // namely the varialbes for which we solve a PDE system

  const SG_EOS<> EOS_phase1; // Equation of state of phase 1
  const SG_EOS<> EOS_phase2; // Equation of state of phase 2

  #ifdef SULICIU_RELAXATION
    samurai::RelaxationFlux<Field> numerical_flux; // function to compute the numerical flux
                                                   // (this is necessary to call 'make_flux')
  #elifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> numerical_flux_cons; // function to compute the numerical flux for the conservative part
                                                     // (this is necessary to call 'make_flux')

    samurai::NonConservativeFlux<Field> numerical_flux_non_cons; // function to compute the numerical flux for the non-conservative part
                                                                 // (this is necessary to call 'make_flux')
  #endif

  // Now we declare a bunch of fields which depend from the state, but it is useful
  // to have it for the output
  Field_Scalar rho,
               p,
               rho1,
               p1,
               c1,
               e1_0,
               e1,
               de1,
               rho2,
               p2,
               c2,
               alpha2,
               Y2,
               e2_0,
               e2,
               de2;

  Field_Vect vel1,
             vel2,
             vel;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void init_variables(); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  void update_auxiliary_fields(); // Routine to update auxilairy fields for output and time step update

  #ifdef RUSANOV_FLUX
    double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue
  #endif

  void update_velocity_before_relaxation(); // Update velocity fields before relaxation

  void apply_instantaneous_velocity_relaxation(); // Apply an instantaneous velocity relaxtion

  void update_pressure_before_relaxation(); // Update pressure fields before relaxation

  void apply_instantaneous_pressure_relaxation(); // Apply an instantaneous pressure relaxation

  #ifdef RELAX_POLYNOM
    void apply_instantaneous_pressure_relaxation_polynomial(); // Apply an instantaneous pressure relaxation (polynomial method Saurel based)
  #endif
};

// Implement class constructor
//
template<std::size_t dim>
Relaxation<dim>::Relaxation(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                            const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                            std::size_t min_level, std::size_t max_level,
                            double Tf_, double cfl_, std::size_t nfiles_,
                            bool do_vel_relax, bool do_pres_relax,
                            bool do_pres_reinit, bool do_energy_update_phase_1,
                            bool do_preserve_energy):
  box(min_corner, max_corner), mesh(box, min_level, max_level, {false}),
  Tf(Tf_), cfl(cfl_), nfiles(nfiles_),
  apply_velocity_relax(do_vel_relax), apply_pressure_relax(do_pres_relax),
  #ifdef RELAX_POLYNOM
    apply_pressure_reinit(do_pres_reinit), start_energy_update_phase_1(do_energy_update_phase_1),
    preserve_energy(do_preserve_energy),
  #endif
  EOS_phase1(EquationData::gamma_1, EquationData::pi_infty_1, EquationData::q_infty_1),
  EOS_phase2(EquationData::gamma_2, EquationData::pi_infty_2, EquationData::q_infty_2),
  #ifdef SULICIU_RELAXATION
    numerical_flux(EOS_phase1, EOS_phase2)
  #elifdef RUSANOV_FLUX
    numerical_flux_cons(EOS_phase1, EOS_phase2),
    numerical_flux_non_cons(EOS_phase1, EOS_phase2)
  #endif
  {
    if(!apply_velocity_relax) {
      assert(!apply_pressure_relax && "You cannot apply pressure relaxation without applying velocity relaxation");
    }
    if(apply_pressure_relax) {
      assert(apply_velocity_relax && "You cannot apply pressure relaxation without applying velocity relaxation");
    }
    std::cout << "Initializing variables" << std::endl;
    std::cout << std::endl;
    init_variables();
  }

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void Relaxation<dim>::init_variables() {
  // Create conserved and auxiliary fields
  conserved_variables = samurai::make_field<double, EquationData::NVARS>("conserved", mesh);

  rho    = samurai::make_field<double, 1>("rho", mesh);
  p      = samurai::make_field<double, 1>("p", mesh);

  rho1   = samurai::make_field<double, 1>("rho1", mesh);
  p1     = samurai::make_field<double, 1>("p1", mesh);
  c1     = samurai::make_field<double, 1>("c1", mesh);

  rho2   = samurai::make_field<double, 1>("rho2", mesh);
  p2     = samurai::make_field<double, 1>("p2", mesh);
  c2     = samurai::make_field<double, 1>("c2", mesh);

  vel1   = samurai::make_field<double, dim>("vel1", mesh);
  vel2   = samurai::make_field<double, dim>("vel2", mesh);
  vel    = samurai::make_field<double, dim>("vel", mesh);

  alpha2 = samurai::make_field<double, 1>("alpha2", mesh);
  Y2     = samurai::make_field<double, 1>("Y2", mesh);

  e1     = samurai::make_field<double, 1>("e1", mesh);
  e1_0   = samurai::make_field<double, 1>("e1_0", mesh);
  de1    = samurai::make_field<double, 1>("de1", mesh);

  e2     = samurai::make_field<double, 1>("e2", mesh);
  e2_0   = samurai::make_field<double, 1>("e2_0", mesh);
  de2    = samurai::make_field<double, 1>("de2", mesh);

  /*--- Set the initial state ---*/
  const double xd = 0.5;

  // Initialize the fields with a loop over all cells
  const double alpha1L = 1.0 - 1e-2;

  const double vel1L   = -2.0;
  const double p1L     = 1e5;
  const double rho1L   = 1150.0;

  const double alpha2L = 1.0 - alpha1L;

  const double vel2L   = -2.0;
  const double p2L     = 1e5;
  const double rho2L   = 0.63;

  const double alpha1R = alpha1L;

  const double vel1R   = 2.0;
  const double p1R     = 1e5;
  const double rho1R   = 1150.0;

  const double alpha2R = 1.0 - alpha1R;

  const double vel2R   = 2.0;
  const double p2R     = 1e5;
  const double rho2R   = 0.63;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];

                           if(x <= xd) {
                             conserved_variables[cell][ALPHA1_INDEX] = alpha1L;

                             vel1[cell] = vel1L;
                             p1[cell]   = p1L;
                             rho1[cell] = rho1L;

                             vel2[cell] = vel2L;
                             p2[cell]   = p2L;
                             rho2[cell] = rho2L;
                           }
                           else {
                             conserved_variables[cell][ALPHA1_INDEX] = alpha1R;

                             vel1[cell] = vel1R;
                             p1[cell]   = p1R;
                             rho1[cell] = rho1R;

                             vel2[cell] = vel2R;
                             p2[cell]   = p2R;
                             rho2[cell] = rho2R;
                           }

                           conserved_variables[cell][ALPHA1_RHO1_INDEX]    = conserved_variables[cell][ALPHA1_INDEX]*rho1[cell];
                           conserved_variables[cell][ALPHA1_RHO1_U1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*vel1[cell];
                           e1[cell] = EOS_phase1.e_value(rho1[cell], p1[cell]);
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*(e1[cell] + 0.5*vel1[cell]*vel1[cell]);

                           conserved_variables[cell][ALPHA2_RHO2_INDEX]    = (1.0 - conserved_variables[cell][ALPHA1_INDEX])*rho2[cell];
                           conserved_variables[cell][ALPHA2_RHO2_U2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*vel2[cell];
                           e2[cell] = EOS_phase2.e_value(rho2[cell], p2[cell]);
                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*(e2[cell] + 0.5*vel2[cell]*vel2[cell]);

                           c1[cell] = EOS_phase1.c_value(rho1[cell], p1[cell]);

                           c2[cell] = EOS_phase2.c_value(rho2[cell], p2[cell]);

                           rho[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           p[cell] = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                   + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*p2[cell];

                           vel[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]/rho[cell]*vel1[cell]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho[cell]*vel2[cell];

                           alpha2[cell] = 1.0 - conserved_variables[cell][ALPHA1_INDEX];
                           Y2[cell]     = conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho[cell];
                         });


  const xt::xtensor_fixed<int, xt::xshape<1>> left{-1};
  const xt::xtensor_fixed<int, xt::xshape<1>> right{1};
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables, alpha1L, alpha1L*rho1L, alpha1L*rho1L*vel1L, alpha1L*rho1L*(EOS_phase1.e_value(rho1L, p1L) + 0.5*vel1L*vel1L),
                                                                        alpha2L*rho2L, alpha2L*rho2L*vel2L, alpha2L*rho2L*(EOS_phase2.e_value(rho2L, p2L) + 0.5*vel2L*vel2L))->on(left);
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables, alpha1R, alpha1R*rho1R, alpha1R*rho1R*vel1R, alpha1R*rho1R*(EOS_phase1.e_value(rho1R, p1R) + 0.5*vel1R*vel1R),
                                                                        alpha2R*rho2R, alpha2R*rho2R*vel2R, alpha2R*rho2R*(EOS_phase2.e_value(rho2R, p2R) + 0.5*vel2R*vel2R))->on(right);
}

// Update velocity fields before relaxation
//
template<std::size_t dim>
void Relaxation<dim>::update_velocity_before_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           vel1[cell] = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX]/conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                                  vanishing volume fraction ---*/
                           vel2[cell] = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX]/conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                                  vanishing volume fraction ---*/
                         });
}

// Apply the instantaneous relaxation for the velocity
//
template<std::size_t dim>
void Relaxation<dim>::apply_instantaneous_velocity_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Compute mixture density and (specific) total energy for the updates
                           const auto rho_0 = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                            + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                           const auto rhoE_0 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]
                                             + conserved_variables[cell][ALPHA2_RHO2_E2_INDEX];

                           // Save specific internal energy of phase 1 for the total energy update
                           auto e1 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                               vanishing volume fraction ---*/
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             e1 -= 0.5*(conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d]/conserved_variables[cell][ALPHA1_RHO1_INDEX])*
                                       (conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d]/conserved_variables[cell][ALPHA1_RHO1_INDEX]); /*--- TODO: Add treatment for
                                                                                                                                                       vanishing volume fraction ---*/
                           }

                           // Update the momentum (and the kinetic energy of phase 1)
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = 0.0;
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             // Compute the equilibrium velocity
                             const auto vel_star_d = (conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d] +
                                                      conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d])/rho_0;

                             // Update the momentum
                             conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*vel_star_d;

                             conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*vel_star_d;

                             // Update the kinetic energy of phase 1
                             conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] += 0.5*conserved_variables[cell][ALPHA1_RHO1_INDEX]*vel_star_d*vel_star_d;
                           }

                           // Update total energy of the two phases
                           const auto Y2      = conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho_0;
                           const auto chi     = 0.0; // uI = (1 - chi)*u1 + chi*u2;
                           const auto e1_star = e1 + 0.5*chi*(vel1[cell] - vel2[cell])*(vel1[cell] - vel2[cell])*Y2;
                           conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] += conserved_variables[cell][ALPHA1_RHO1_INDEX]*e1_star;

                           conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = rhoE_0 - conserved_variables[cell][ALPHA1_RHO1_E1_INDEX];
                         });
}

// Update pressure fields before relaxation
//
template<std::size_t dim>
void Relaxation<dim>::update_pressure_before_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           e1_0[cell] = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                                  vanishing volume fraction ---*/

                           e2_0[cell] = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                                  vanishing volume fraction ---*/
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             const auto vel1_d = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d]/conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                                               vanishing volume fraction ---*/
                             e1_0[cell] -= 0.5*vel1_d*vel1_d;

                             const auto vel2_d = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d]/conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                                               vanishing volume fraction ---*/
                             e2_0[cell] -= 0.5*vel2_d*vel2_d;
                           }
                           p1[cell] = EOS_phase1.pres_value(conserved_variables[cell][ALPHA1_RHO1_INDEX]/conserved_variables[cell][ALPHA1_INDEX], e1_0[cell]);
                           /*--- TODO: Add treatment for vanishing volume fraction ---*/

                           p2[cell] = EOS_phase2.pres_value(conserved_variables[cell][ALPHA2_RHO2_INDEX]/(1.0 - conserved_variables[cell][ALPHA1_INDEX]), e2_0[cell]);
                           /*--- TODO: Add treatment for vanishing volume fraction ---*/
                         });
}

// Apply the instantaneous relaxation for the pressure
//
template<std::size_t dim>
void Relaxation<dim>::apply_instantaneous_pressure_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                            /*--- Save some quantities which remain constant during relaxation
                                  for the sake of convenience and readability ---*/
                            const auto rhoE_0 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]
                                              + conserved_variables[cell][ALPHA2_RHO2_E2_INDEX];

                            const auto arho1_0 = conserved_variables[cell][ALPHA1_RHO1_INDEX];
                            const auto arho2_0 = conserved_variables[cell][ALPHA2_RHO2_INDEX];

                            const auto rho_0 = arho1_0 + arho2_0;

                            auto vel_squared = 0.0;
                            for(std::size_t d = 0; d < EquationData::dim; ++d) {
                              const auto vel_d = (conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d] +
                                                  conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d])/rho_0;
                              vel_squared += vel_d*vel_d;
                            }

                            const auto e_0 = rhoE_0/rho_0 - 0.5*vel_squared;

                            const auto Y1_0 = arho1_0/rho_0;
                            const auto Y2_0 = 1.0 - Y1_0;

                            const auto gammaM1_1 = EquationData::gamma_1 - 1.0;
                            const auto gammaM1_2 = EquationData::gamma_2 - 1.0;

                            const auto gampinf_1 = EquationData::gamma_1 * EquationData::pi_infty_1;
                            const auto gampinf_2 = EquationData::gamma_2 * EquationData::pi_infty_2;

                            /*--- Take interface pressure equal to the liquid one (for the moment) ---*/
                            auto pres1 = p1[cell];
                            auto pres2 = p2[cell];
                            auto& pI   = pres1;

                            const auto Laplace_cst_1 = (pres1 + EquationData::pi_infty_1)/
                                                       std::pow(arho1_0/conserved_variables[cell][ALPHA1_INDEX], EquationData::gamma_1);
                            //const auto Laplace_cst_2 = (pres2 + EquationData::pi_infty_2)/
                            //                           std::pow(arho2_0/(1.0 - conserved_variables[cell][ALPHA1_INDEX]), EquationData::gamma_2);

                            /*--- Newton method to compute the new volume fraction ---*/
                            const double tol             = 1e-8;
                            const double lambda          = 0.9; // Bound preserving parameter
                            const unsigned int max_iters = 100;
                            auto alpha_max               = 1.0;
                            auto alpha_min               = 0.0;

                            auto dalpha1      = 0.0;
                            unsigned int nite = 0;
                            while (nite < max_iters && 2.0*(alpha_max - alpha_min)/(alpha_max + alpha_min) > tol) {
                              pres1 > pres2 ? alpha_min = conserved_variables[cell][ALPHA1_INDEX] :
                                              alpha_max = conserved_variables[cell][ALPHA1_INDEX];

                              dalpha1 = (pres1 - pres2)/
                                        std::abs((pres1 + gammaM1_1*pI + gampinf_1)/conserved_variables[cell][ALPHA1_INDEX] +
                                                 (pres2 + gammaM1_2*pI + gampinf_2)/(1.0 - conserved_variables[cell][ALPHA1_INDEX]));

                              // Bound preserving strategy
                              dalpha1 = std::min(dalpha1, lambda*(alpha_max - conserved_variables[cell][ALPHA1_INDEX]));
                              dalpha1 = std::max(dalpha1, lambda*(alpha_min - conserved_variables[cell][ALPHA1_INDEX]));

                              conserved_variables[cell][ALPHA1_INDEX] += dalpha1;

                              // Update pressure variables for next step and to update interfacial pressure
                              const auto rho1 = arho1_0/conserved_variables[cell][ALPHA1_INDEX];
                              const auto rho2 = arho2_0/(1.0 - conserved_variables[cell][ALPHA1_INDEX]);

                              pres1         = std::pow(rho1, EquationData::gamma_1)*Laplace_cst_1 - EquationData::pi_infty_1;
                              const auto e2 = (e_0 - Y1_0*EOS_phase1.e_value(rho1, pres1))/Y2_0;
                              pres2         = EOS_phase2.pres_value(rho2, e2);

                              /*pres2         = std::pow(rho2, EquationData::gamma_2)*Laplace_cst_2 - EquationData::pi_infty_2;
                              const auto e1 = (e_0 - Y2_0*EOS_phase2.e_value(rho2, pres2))/Y1_0;
                              pres1         = EOS_phase1.pres_value(rho1, e1);*/

                              nite++;
                            }

                            conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = arho1_0*(EOS_phase1.e_value(arho1_0/conserved_variables[cell][ALPHA1_INDEX], pres1) + 0.5*vel_squared);
                            conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = rhoE_0 - conserved_variables[cell][ALPHA1_RHO1_E1_INDEX];
                          });
}


// Apply the instantaneous relaxation for the pressure (polynomial method Saurel)
//
#ifdef RELAX_POLYNOM
template<std::size_t dim>
void Relaxation<dim>::apply_instantaneous_pressure_relaxation_polynomial() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Save mixture total energy for later update
                           const auto rhoE_0 = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]
                                             + conserved_variables[cell][ALPHA2_RHO2_E2_INDEX];

                           // Compute the pressure equilibirum with the polynomial method (Saurel)
                           const auto b = (conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2*(EquationData::pi_infty_2 - p1[cell]) +
                                           (1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1*(EquationData::pi_infty_1 - p2[cell]))/
                                          (conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2 + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1);
                           const auto c = -(p1[cell]*conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2*EquationData::pi_infty_2 +
                                            p2[cell]*(1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1*EquationData::pi_infty_1)/
                                           (conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_2 + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_1);

                           auto p_star = 0.5*(-b + std::sqrt(b*b - 4.0*c));

                           // Update the volume fraction using the computed pressure
                           conserved_variables[cell][ALPHA1_INDEX] *= (p1[cell] + EquationData::gamma_1*EquationData::pi_infty_1 + p_star*(EquationData::gamma_1 - 1.0))/
                                                                      (p_star + EquationData::gamma_1*EquationData::pi_infty_1 + p_star*(EquationData::gamma_1 - 1.0));

                           // Apply the pressure reinitialization if desired
                           if(apply_pressure_reinit) {
                             const auto rho_0 = conserved_variables[cell][ALPHA1_RHO1_INDEX]
                                              + conserved_variables[cell][ALPHA2_RHO2_INDEX];

                             auto rhoe_0 = rhoE_0;
                             for(std::size_t d = 0; d < EquationData::dim; ++d) {
                               // Compute the equilibrium velocity
                               // (use the mixture law, so as to avoid divsion by zero, even though it is the same for each phase)
                               const auto vel_star_d = (conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d] +
                                                        conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d])/rho_0;
                               rhoe_0 -= 0.5*rho_0*vel_star_d*vel_star_d;
                             }
                             p_star = (rhoe_0 -
                                       (conserved_variables[cell][ALPHA1_RHO1_INDEX]*EquationData::q_infty_1 +
                                        conserved_variables[cell][ALPHA2_RHO2_INDEX]*EquationData::q_infty_2) -
                                       (conserved_variables[cell][ALPHA1_INDEX]*EquationData::gamma_1*EquationData::pi_infty_1/(EquationData::gamma_1 - 1.0) +
                                        (1.0 - conserved_variables[cell][ALPHA1_INDEX])*EquationData::gamma_2*EquationData::pi_infty_2/(EquationData::gamma_2 - 1.0)))/
                                      (conserved_variables[cell][ALPHA1_INDEX]/(EquationData::gamma_1 - 1.0) +
                                       (1.0 - conserved_variables[cell][ALPHA1_INDEX])/(EquationData::gamma_2 - 1.0));
                           }

                           // Update the total energy starting from phase 1
                           if(start_energy_update_phase_1) {
                             // Update the total energy of phase 1
                             auto E1 = EOS_phase1.e_value(conserved_variables[cell][ALPHA1_RHO1_INDEX]/conserved_variables[cell][ALPHA1_INDEX], p_star);
                             /*--- TODO: Add treatment for vanishing volume fraction ---*/

                             for(std::size_t d = 0; d < EquationData::dim; ++d) {
                               const auto vel1_d = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d]/conserved_variables[cell][ALPHA1_RHO1_INDEX];
                               /*--- TODO: Add treatment for vanishing volume fraction ---*/

                               E1 += 0.5*vel1_d*vel1_d;
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
                                 const auto vel2_d = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d]/conserved_variables[cell][ALPHA2_RHO2_INDEX];
                                 /*--- TODO: Add treatment for vanishing volume fraction ---*/

                                 E2 += 0.5*vel2_d*vel2_d;
                               }

                               conserved_variables[cell][ALPHA2_RHO2_E2_INDEX] = conserved_variables[cell][ALPHA2_RHO2_INDEX]*E2;
                             }
                           }
                           // Update the total energy starting from phase 2
                           else {
                             // Update the total energy of phase 2
                             auto E2 = EOS_phase2.e_value(conserved_variables[cell][ALPHA2_RHO2_INDEX]/(1.0 - conserved_variables[cell][ALPHA1_INDEX]), p_star);
                             /*--- TODO: Add treatment for vanishing volume fraction ---*/

                             for(std::size_t d = 0; d < EquationData::dim; ++d) {
                               const auto vel2_d = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX + d]/conserved_variables[cell][ALPHA2_RHO2_INDEX];
                               /*--- TODO: Add treatment for vanishing volume fraction ---*/

                               E2 += 0.5*vel2_d*vel2_d;
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
                                 const auto vel1_d = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX + d]/conserved_variables[cell][ALPHA1_RHO1_INDEX];
                                 /*--- TODO: Add treatment for vanishing volume fraction ---*/

                                 E1 += 0.5*vel1_d*vel1_d;
                               }

                               conserved_variables[cell][ALPHA1_RHO1_E1_INDEX] = conserved_variables[cell][ALPHA1_RHO1_INDEX]*E1;
                             }
                           }
                         });
}
#endif

#ifdef RUSANOV_FLUX
  // Compute the estimate of the maximum eigenvalue for CFL condition
  //
  template<std::size_t dim>
  double Relaxation<dim>::get_max_lambda() const {
    double res = 0.0;

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             res = std::max(std::max(std::abs(vel1[cell]) + c1[cell],
                                                     std::abs(vel2[cell]) + c2[cell]),
                                            res);
                           });

    return res;
  }
#endif

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
                           vel1[cell] = conserved_variables[cell][ALPHA1_RHO1_U1_INDEX]/conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                                  vanishing volume fraction ---*/
                           e1[cell] = conserved_variables[cell][ALPHA1_RHO1_E1_INDEX]/conserved_variables[cell][ALPHA1_RHO1_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                                vanishing volume fraction ---*/
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             e1[cell] -= 0.5*vel1[cell]*vel1[cell];
                           }
                           de1[cell] = e1[cell] - e1_0[cell];
                           p1[cell]  = EOS_phase1.pres_value(rho1[cell], e1[cell]);
                           c1[cell]  = EOS_phase1.c_value(rho1[cell], p1[cell]);

                           /*--- Phase 2 ---*/
                           rho2[cell] = conserved_variables[cell][ALPHA2_RHO2_INDEX]/(1.0 - conserved_variables[cell][ALPHA1_INDEX]); /*--- TODO: Add treatment for
                                                                                                                                                  vanishing volume fraction ---*/
                           vel2[cell] = conserved_variables[cell][ALPHA2_RHO2_U2_INDEX]/conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for v
                                                                                                                                                  vanishing volume fraction ---*/
                           e2[cell] = conserved_variables[cell][ALPHA2_RHO2_E2_INDEX]/conserved_variables[cell][ALPHA2_RHO2_INDEX]; /*--- TODO: Add treatment for
                                                                                                                                                vanishing volume fraction ---*/
                           for(std::size_t d = 0; d < EquationData::dim; ++d) {
                             e2[cell] -= 0.5*vel2[cell]*vel2[cell];
                           }
                           de2[cell] = e2[cell] - e2_0[cell];
                           p2[cell]  = EOS_phase2.pres_value(rho2[cell], e2[cell]);
                           c2[cell]  = EOS_phase2.c_value(rho2[cell], p2[cell]);

                           alpha2[cell] = 1.0 - conserved_variables[cell][ALPHA1_INDEX];
                           Y2[cell]     = conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho[cell];

                           /*--- Remaining mixture quantities ---*/
                           p[cell] = conserved_variables[cell][ALPHA1_INDEX]*p1[cell]
                                   + (1.0 - conserved_variables[cell][ALPHA1_INDEX])*p2[cell];

                           vel[cell] = conserved_variables[cell][ALPHA1_RHO1_INDEX]/rho[cell]*vel1[cell]
                                     + conserved_variables[cell][ALPHA2_RHO2_INDEX]/rho[cell]*vel2[cell];
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
  fs::path path = fs::current_path();
  #ifdef SULICIU_RELAXATION
    std::string filename = "Relaxation_Suliciu";
  #elifdef RUSANOV_FLUX
    std::string filename = "Relaxation_Rusanov";
  #endif

  #ifdef ORDER_2
    filename = filename + "_order2";
  #else
    filename = filename + "_order1";
  #endif

  const double dt_save = Tf/static_cast<double>(nfiles);

  /*--- Auxiliary variables to save updated fields ---*/
  #ifdef ORDER_2
    auto conserved_variables_tmp   = samurai::make_field<double, EquationData::NVARS>("conserved_tmp", mesh);
    auto conserved_variables_tmp_2 = samurai::make_field<double, EquationData::NVARS>("conserved_tmp_2", mesh);
  #endif
  auto conserved_variables_np1 = samurai::make_field<double, EquationData::NVARS>("conserved_np1", mesh);

  /*--- Create the flux variables ---*/
  #ifdef SULICIU_RELAXATION
    double c = 0.0;
    auto Suliciu_flux = numerical_flux.make_flux(c);
  #elifdef RUSANOV_FLUX
    auto Rusanov_flux         = numerical_flux_cons.make_flux();
    auto NonConservative_flux = numerical_flux_non_cons.make_flux();
  #endif

  /*--- Save the initial condition ---*/
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, conserved_variables, rho, p, vel, vel1, rho1, p1, c1, e1, vel2, rho2, p2, c2, alpha2, Y2, e2);

  /*--- Set mesh size ---*/
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  const double dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());

  /*--- Start the loop ---*/
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  double t          = 0.0;
  while(t != Tf) {
    // Apply the numerical scheme
    samurai::update_ghost_mr(conserved_variables);
    samurai::update_bc(conserved_variables);

    #ifdef SULICIU_RELAXATION
      c = 0.0;
      auto Relaxation_Flux = Suliciu_flux(conserved_variables);
      const double dt = std::min(Tf - t, cfl*dx/c);
      t += dt;
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

      #ifdef ORDER_2
        conserved_variables_tmp = conserved_variables - dt*Relaxation_Flux;
      #else
        conserved_variables_np1 = conserved_variables - dt*Relaxation_Flux;
      #endif
    #elifdef RUSANOV_FLUX
      auto Cons_Flux    = Rusanov_flux(conserved_variables);
      auto NonCons_Flux = NonConservative_flux(conserved_variables);
      if(nt > 0 && !(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)) {
        update_auxiliary_fields();
      }
      const double dt = std::min(Tf - t, cfl*dx/get_max_lambda());
      t += dt;
      std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

      #ifdef ORDER_2
        conserved_variables_tmp = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #else
        conserved_variables_np1 = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #endif
    #endif

    #ifdef ORDER_2
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());
    #else
      std::swap(conserved_variables.array(), conserved_variables_np1.array());
    #endif

    // Apply the relaxation for the velocity
    if(apply_velocity_relax) {
      update_velocity_before_relaxation();
      apply_instantaneous_velocity_relaxation();

      if(apply_pressure_relax) {
        // Apply the relaxation for the pressure
        update_pressure_before_relaxation();
        #ifdef RELAX_POLYNOM
          apply_instantaneous_pressure_relaxation_polynomial();
        #else
          apply_instantaneous_pressure_relaxation();
        #endif
      }
    }

    // Consider the second stage for second order
    #ifdef ORDER_2
      samurai::update_ghost_mr(conserved_variables);
      samurai::update_bc(conserved_variables);

      #ifdef SULICIU_RELAXATION
        Relaxation_Flux = Suliciu_flux(conserved_variables);

        conserved_variables_tmp_2 = conserved_variables - dt*Relaxation_Flux;
      #elifdef RUSANOV_FLUX
        Cons_Flux    = Rusanov_flux(conserved_variables);
        NonCons_Flux = NonConservative_flux(conserved_variables);

        conserved_variables_tmp_2 = conserved_variables - dt*Cons_Flux - dt*NonCons_Flux;
      #endif
      conserved_variables_np1 = 0.5*(conserved_variables_tmp + conserved_variables_tmp_2);
      std::swap(conserved_variables.array(), conserved_variables_np1.array());

      // Apply the relaxation for the velocity
      if(apply_velocity_relax) {
        update_velocity_before_relaxation();
        apply_instantaneous_velocity_relaxation();

        if(apply_pressure_relax) {
          // Apply the relaxation for the pressure
          update_pressure_before_relaxation();
          #ifdef RELAX_POLYNOM
            apply_instantaneous_pressure_relaxation_polynomial();
          #else
            apply_instantaneous_pressure_relaxation();
          #endif
        }
      }
    #endif

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      update_auxiliary_fields();

      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix, conserved_variables, rho, p, vel, vel1, rho1, p1, c1, e1_0, e1, de1, vel2, rho2, p2, c2, alpha2, Y2, e2_0, e2, de2);
    }
  }
}
