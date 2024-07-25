// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <numbers>

#include <filesystem>
namespace fs = std::filesystem;

#include "barotropic_eos.hpp"

#include "waves_interface_FV.hpp"

// Add header file for the multiresolution
#include <samurai/mr/adapt.hpp>

// Auxiliary function to compute the regualized Heaviside
template<typename T = double>
T CHeaviside(const T x, const T eps) {
  if(x < -eps) {
    return 0.0;
  }
  else if(x > eps) {
    return 1.0;
  }

  const double pi = 4.0*std::atan(1);
  return 0.5*(1.0 + x/eps + 1.0/pi*std::sin(pi*x/eps));
}

#define GODUNOV_FLUX
//#define RUSANOV_FLUX

// Specify the use of this namespace where we just store the indices
// and, in this case, some parameters related to EOS
using namespace EquationData;

// This is the class for the simulation of a two-scale model
// for the waves-interface interaction
//
template<std::size_t dim>
class WaveInterface {
public:
  using Config = samurai::MRConfig<dim, 2, 2, 2>;

  WaveInterface() = default; // Default constructor. This will do nothing
                             // and basically will never be used

  WaveInterface(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                std::size_t min_level, std::size_t max_level,
                double Tf_, double cfl_, std::size_t nfiles_ = 100,
                bool apply_relax_ = true);                             // Class constrcutor with the arguments related
                                                                       // to the grid, to the physics and to the relaxation.
                                                                       // Maybe in the future, we could think to add parameters related to EOS

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
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;

  using Field        = samurai::Field<decltype(mesh), double, EquationData::NVARS, false>;
  using Field_Scalar = samurai::Field<decltype(mesh), double, 1, false>;
  using Field_Vect   = samurai::Field<decltype(mesh), double, dim, false>;

  bool apply_relax; // Choose whether to apply or not the relaxation

  double Tf;  // Final time of the simulation
  double cfl; // Courant number of the simulation so as to compute the time step

  std::size_t nfiles; // Number of files desired for output

  Field conserved_variables; // The variable which stores the conserved variables,
                             // namely the varialbes for which we solve a PDE system

  // Now we declare a bunch of fields which depend from the state, but it is useful
  // to have it so as to avoid recomputation
  Field_Scalar alpha1,
               dalpha1,
               p1,
               p2,
               p,
               rho;

  Field_Vect   u;

  double eps; // Tolerance when we want to avoid division by zero

  LinearizedBarotropicEOS<> EOS_phase1,
                            EOS_phase2; // The two variables which take care of the
                                        // barotropic EOS to compute the speed of sound
  #ifdef RUSANOV_FLUX
    samurai::RusanovFlux<Field> Rusanov_flux; // Auxiliary variable to compute the flux
  #elifdef GODUNOV_FLUX
    samurai::GodunovFlux<Field> Godunov_flux; // Auxiliary variable to compute the flux
  #endif

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void init_variables(); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue

  void apply_relaxation(); // Apply the relaxation
};

// Implement class constructor
//
#ifdef RUSANOV_FLUX
  template<std::size_t dim>
  WaveInterface<dim>::WaveInterface(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                    const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                    std::size_t min_level, std::size_t max_level,
                                    double Tf_, double cfl_, std::size_t nfiles_,
                                    bool apply_relax_):
    box(min_corner, max_corner), mesh(box, min_level, max_level, {false}),
    apply_relax(apply_relax_), Tf(Tf_), cfl(cfl_), nfiles(nfiles_),
    eps(1e-20),
    EOS_phase1(EquationData::p0_phase1, EquationData::rho0_phase1, EquationData::c0_phase1),
    EOS_phase2(EquationData::p0_phase2, EquationData::rho0_phase2, EquationData::c0_phase2),
    Rusanov_flux(EOS_phase1, EOS_phase2, eps) {
      std::cout << "Initializing variables " << std::endl;
      std::cout << std::endl;
      init_variables();
  }
#elifdef GODUNOV_FLUX
  template<std::size_t dim>
  WaveInterface<dim>::WaveInterface(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                    const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                    std::size_t min_level, std::size_t max_level,
                                    double Tf_, double cfl_, std::size_t nfiles_,
                                    bool apply_relax_):
    box(min_corner, max_corner), mesh(box, min_level, max_level, {false}),
    apply_relax(apply_relax_), Tf(Tf_), cfl(cfl_), nfiles(nfiles_),
    eps(1e-20),
    EOS_phase1(EquationData::p0_phase1, EquationData::rho0_phase1, EquationData::c0_phase1),
    EOS_phase2(EquationData::p0_phase2, EquationData::rho0_phase2, EquationData::c0_phase2),
    Godunov_flux(EOS_phase1, EOS_phase2, eps) {
      std::cout << "Initializing variables " << std::endl;
      std::cout << std::endl;
      init_variables();
  }
#endif

// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void WaveInterface<dim>::init_variables() {
  // Create conserved and auxiliary fields
  conserved_variables = samurai::make_field<double, EquationData::NVARS>("conserved", mesh);

  alpha1  = samurai::make_field<double, 1>("alpha1", mesh);

  dalpha1 = samurai::make_field<double, 1>("dalpha1", mesh);

  p1      = samurai::make_field<double, 1>("p1", mesh);
  p2      = samurai::make_field<double, 1>("p2", mesh);
  p       = samurai::make_field<double, 1>("p", mesh);
  rho     = samurai::make_field<double, 1>("rho", mesh);

  u       = samurai::make_field<double, dim>("u", mesh);

  // Declare some constant parameters associated to the grid and to the
  // initial state
  const double x_shock       = 0.3;
  const double x_interface   = 0.7;
  const double dx            = samurai::cell_length(mesh[mesh_id_t::cells].max_level());
  const double eps_interface = 3.0*dx;
  const double eps_shock     = 3.0*dx;

  // Initialize some fields to define the bubble with a loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];

                           // Set volume fraction
                           alpha1[cell] = 1e-7 + (1.0 - 2e-7)*CHeaviside(x_interface - x, eps_interface)
                                               + (0.999999997719987 - 1.0 + 1e-7)*CHeaviside(x_shock - x , eps_shock);

                           // Set mass phase 1
                           const double rho1 = 1000.0 + (1001.857557720546 - 1000.0)*CHeaviside(x_shock - x, eps_shock);
                           p1[cell] = EOS_phase1.pres_value(rho1);

                           conserved_variables[cell][M1_INDEX] = alpha1[cell]*rho1;

                           // Set mass phase 2
                           const double rho2 = 1.0 + (43.77807526718601 - 1.0)*CHeaviside(x_shock - x, eps_shock);
                           p2[cell] = EOS_phase2.pres_value(rho2);

                           conserved_variables[cell][M2_INDEX] = (1.0 - alpha1[cell])*rho2;

                           // Set conserved variable associated to volume fraction
                           rho[cell] = conserved_variables[cell][M1_INDEX]
                                     + conserved_variables[cell][M2_INDEX];

                           conserved_variables[cell][RHO_ALPHA1_INDEX] = rho[cell]*alpha1[cell];

                           // Set momentum
                           u[cell] = 3.0281722661268375*CHeaviside(x_shock - x, eps_shock);
                           conserved_variables[cell][RHO_U_INDEX] = rho[cell]*u[cell];

                           // Set mixture pressure for output
                           p[cell] = alpha1[cell]*p1[cell] + (1.0 - alpha1[cell])*p2[cell];
                         });

  // Consider Neumann bcs
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables, 0.0, 0.0, 0.0, 0.0);
}

// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double WaveInterface<dim>::get_max_lambda() const {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Compute the velocity along both horizontal and vertical direction
                           const auto vel_x = conserved_variables[cell][RHO_U_INDEX]/rho[cell];

                           // Compute frozen speed of sound
                           const auto rho1      = (alpha1[cell] > eps) ? conserved_variables[cell][M1_INDEX]/alpha1[cell] : nan("");
                           const auto alpha2    = 1.0 - alpha1[cell];
                           const auto rho2      = (alpha2 > eps) ? conserved_variables[cell][M2_INDEX]/alpha2 : nan("");
                           const auto c_squared = conserved_variables[cell][M1_INDEX]*EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                                + conserved_variables[cell][M2_INDEX]*EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);
                           const auto c         = std::sqrt(c_squared/rho[cell]);

                           // Update eigenvalue estimate
                           res = std::max(std::abs(vel_x) + c, res);
                         });

  return res;
}

// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void WaveInterface<dim>::apply_relaxation() {
  const double tol    = 1e-12; /*--- Tolerance of the Newton method ---*/
  const double lambda = 0.9;   /*--- Parameter for bound preserving strategy ---*/

  // Loop of Newton method. Conceptually, a loop over cells followed by a Newton loop
  // over each cell would be more logic, but this would lead to issues to call 'update_geometry'
  std::size_t Newton_iter = 0;
  bool relaxation_applied = true;
  while(relaxation_applied == true) {
    relaxation_applied = false;
    Newton_iter++;

    // Loop over all cells.
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             #ifdef RUSANOV_FLUX
                               Rusanov_flux.perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                                           dalpha1[cell], alpha1[cell], rho[cell], relaxation_applied, tol, lambda);
                             #elifdef GODUNOV_FLUX
                               Godunov_flux.perform_Newton_step_relaxation(std::make_unique<decltype(conserved_variables[cell])>(conserved_variables[cell]),
                                                                           dalpha1[cell], alpha1[cell], rho[cell], relaxation_applied, tol, lambda);
                             #endif

                           });

    // Newton cycle diverged
    if(Newton_iter > 60) {
      std::cout << "Netwon method not converged in the post-hyperbolic relaxation" << std::endl;
      save(fs::current_path(), "waves_interface", "_diverged",
           conserved_variables, alpha1, rho);
      exit(1);
    }
  }
}

// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void WaveInterface<dim>::save(const fs::path& path,
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
void WaveInterface<dim>::run() {
  // Default output arguemnts
  fs::path path        = fs::current_path();
  #ifdef RUSANOV_FLUX
    std::string filename = "waves_interface_Rusanov";
  #elifdef GODUNOV_FLUX
    std::string filename = "waves_interface_Godunov";
  #endif

  #ifdef ORDER_2
    filename = filename + "_order2";
  #else
    filename = filename + "_order1";
  #endif

  const double dt_save = Tf/static_cast<double>(nfiles);

  // Auxiliary variable to save updated fields
  #ifdef ORDER_2
    auto conserved_variables_tmp   = samurai::make_field<double, EquationData::NVARS>("conserved_tmp", mesh);
    auto conserved_variables_tmp_2 = samurai::make_field<double, EquationData::NVARS>("conserved_tmp_2", mesh);
  #endif
  auto conserved_variables_np1   = samurai::make_field<double, EquationData::NVARS>("conserved_np1", mesh);

  // Create the flux variable
  #ifdef RUSANOV_FLUX
    auto numerical_flux = Rusanov_flux.make_flux();
  #elifdef GODUNOV_FLUX
    auto numerical_flux = Godunov_flux.make_flux();
  #endif

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, conserved_variables, alpha1, rho, p1, p2, p, u);

  // Set initial time step
  const double dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());
  double dt       = cfl*dx/get_max_lambda();

  // Start the loop
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  double t          = 0.0;
  while(t != Tf) {
    t += dt;
    if(t > Tf) {
      dt += Tf - t;
      t = Tf;
    }

    std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

    /*--- Apply mesh adaptation ---*/
    samurai::update_ghost_mr(alpha1);
    auto MRadaptation = samurai::make_MRAdapt(alpha1);
    MRadaptation(1e-5, 0, conserved_variables);

    /*--- Apply the numerical scheme without relaxation ---*/
    samurai::update_ghost_mr(conserved_variables);
    auto flux_conserved = numerical_flux(conserved_variables);
    #ifdef ORDER_2
      conserved_variables_tmp.resize();
      conserved_variables_tmp = conserved_variables - dt*flux_conserved;
      std::swap(conserved_variables.array(), conserved_variables_tmp.array());
    #else
      conserved_variables_np1.resize();
      conserved_variables_np1 = conserved_variables - dt*flux_conserved;
      std::swap(conserved_variables.array(), conserved_variables_np1.array());
    #endif

    // Sanity check (and numerical artefacts to clear data) after hyperbolic step
    // and before Newton loop (if desired).
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             // Start with rho_alpha1
                             if(conserved_variables[cell][RHO_ALPHA1_INDEX] < 0.0) {
                               if(conserved_variables[cell][RHO_ALPHA1_INDEX] < -1e-10) {
                                 std::cerr << " Negative volume fraction at the beginning of the relaxation" << std::endl;
                                 save(fs::current_path(), "waves_interface", "_diverged", conserved_variables);
                                 exit(1);
                               }
                               conserved_variables[cell][RHO_ALPHA1_INDEX] = 0.0;
                             }
                             // Sanity check for m1
                             if(conserved_variables[cell][M1_INDEX] < 0.0) {
                               if(conserved_variables[cell][M1_INDEX] < -1e-14) {
                                 std::cerr << "Negative mass for phase 1 at the beginning of the relaxation" << std::endl;
                                 save(fs::current_path(), "waves_interface", "_diverged", conserved_variables);
                                 exit(1);
                               }
                               conserved_variables[cell][M1_INDEX] = 0.0;
                             }
                             // Sanity check for m2
                             if(conserved_variables[cell][M2_INDEX] < 0.0) {
                               if(conserved_variables[cell][M2_INDEX] < -1e-14) {
                                 std::cerr << "Negative mass for phase 2 at the beginning of the relaxation" << std::endl;
                                 save(fs::current_path(), "waves_interface", "_diverged", conserved_variables);
                                 exit(1);
                               }
                               conserved_variables[cell][M2_INDEX] = 0.0;
                             }

                             // Update volume fraction (and consequently density)
                             rho.resize();
                             rho[cell]    = conserved_variables[cell][M1_INDEX]
                                          + conserved_variables[cell][M2_INDEX];

                             alpha1[cell] = std::min(std::max(0.0, conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell]), 1.0);
                           });

    /*--- Apply relaxation ---*/
    if(apply_relax) {
      // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
      // concerns next time step, rho_alpha1
      dalpha1.resize();
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               dalpha1[cell] = std::numeric_limits<double>::infinity();
                             });
      apply_relaxation();
    }

    /*--- Consider the second stage for the second order ---*/
    #ifdef ORDER_2
      // Apply the numerical scheme
      samurai::update_ghost_mr(conserved_variables);
      flux_conserved = numerical_flux(conserved_variables);
      conserved_variables_tmp_2.resize();
      conserved_variables_tmp_2 = conserved_variables - dt*flux_conserved;
      conserved_variables_np1   = 0.5*(conserved_variables_tmp + conserved_variables_tmp_2);
      std::swap(conserved_variables.array(), conserved_variables_np1.array());

      // Sanity check (and numerical artefacts to clear data) after hyperbolic step
      // and before Newton loop (if desired).
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               // Start with rho_alpha1
                               if(conserved_variables[cell][RHO_ALPHA1_INDEX] < 0.0) {
                                 if(conserved_variables[cell][RHO_ALPHA1_INDEX] < -1e-10) {
                                   std::cerr << " Negative volume fraction at the beginning of the relaxation" << std::endl;
                                   save(fs::current_path(), "waves_interface", "_diverged", conserved_variables);
                                   exit(1);
                                 }
                                 conserved_variables[cell][RHO_ALPHA1_INDEX] = 0.0;
                               }
                               // Sanity check for m1
                               if(conserved_variables[cell][M1_INDEX] < 0.0) {
                                 if(conserved_variables[cell][M1_INDEX] < -1e-14) {
                                   std::cerr << "Negative mass for phase 1 at the beginning of the relaxation" << std::endl;
                                   save(fs::current_path(), "waves_interface", "_diverged", conserved_variables);
                                   exit(1);
                                 }
                                 conserved_variables[cell][M1_INDEX] = 0.0;
                               }
                               // Sanity check for m2
                               if(conserved_variables[cell][M2_INDEX] < 0.0) {
                                 if(conserved_variables[cell][M2_INDEX] < -1e-14) {
                                   std::cerr << "Negative mass for phase 2 at the beginning of the relaxation" << std::endl;
                                   save(fs::current_path(), "waves_interface", "_diverged", conserved_variables);
                                   exit(1);
                                 }
                                 conserved_variables[cell][M2_INDEX] = 0.0;
                               }

                               // Update volume fraction (and consequently density)
                               rho[cell]    = conserved_variables[cell][M1_INDEX]
                                            + conserved_variables[cell][M2_INDEX];

                               alpha1[cell] = std::min(std::max(0.0, conserved_variables[cell][RHO_ALPHA1_INDEX]/rho[cell]), 1.0);
                             });

      // Apply the relaxation
      if(apply_relax) {
        // Apply relaxation if desired, which will modify alpha1 and, consequently, for what
        // concerns next time step, rho_alpha1
        samurai::for_each_cell(mesh,
                               [&](const auto& cell)
                               {
                                 dalpha1[cell] = std::numeric_limits<double>::infinity();
                               });
        apply_relaxation();
      }
    #endif

    /*--- Compute updated time step ---*/
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());

    /*--- Save the results ---*/
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";

      // Compute auxliary fields for the output
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               // Compute pressure fields
                               p1.resize();
                               const auto rho1   = (alpha1[cell] > eps) ? conserved_variables[cell][M1_INDEX]/alpha1[cell] : nan("");
                               p1[cell]          = EOS_phase1.pres_value(rho1);

                               p2.resize();
                               const auto alpha2 = 1.0 - alpha1[cell];
                               const auto rho2   = (alpha2 > eps) ? conserved_variables[cell][M2_INDEX]/alpha2 : nan("");
                               p2[cell]          = EOS_phase2.pres_value(rho2);

                               p.resize();
                               p[cell]           = (alpha1[cell] > eps && alpha2 > eps) ?
                                                   alpha1[cell]*p1[cell] + alpha2*p2[cell] :
                                                   ((alpha1[cell] < eps) ? p2[cell] : p1[cell]);

                              // Compute velocity field
                              u.resize();
                              u[cell] = conserved_variables[cell][RHO_U_INDEX]/rho[cell];
                             });

      save(path, filename, suffix, conserved_variables, alpha1, rho, p1, p2, p, u);
    }
  }
}
