// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include <samurai/algorithm/update.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

#include "barotropic_eos.hpp"

#include "two_scale_capillarity_FV.hpp"

// Add header file for the multiresolution
#include <samurai/mr/adapt.hpp>

// Specify the use of this namespace where we just store the indices
// and, in this case, some parameters related to EOS
using namespace EquationData;

// This is the class for the simulation of a two-scale model with capillarity
//
template<std::size_t dim>
class TwoScaleCapillarity {
public:
  using Config = samurai::MRConfig<dim, 2>;

  TwoScaleCapillarity() = default; // Default constructor. This will do nothing
                                   // and basically will never be used

  TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                      const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                      std::size_t min_level, std::size_t max_level,
                      double Tf_, double cfl_, std::size_t nfiles_ = 100,
                      bool apply_relax_ = true, bool mass_transfer_ = false); // Class constrcutor with the arguments related
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

  bool mass_transfer; // Choose wheter to apply or not the mass transfer

  std::size_t nfiles; // Number of files desired for output

  Field conserved_variables; // The variable which stores the conserved variables,
                             // namely the varialbes for which we solve a PDE system

  // Now we declare a bunch of fields which depend from the state, but it is useful
  // to have it so as to avoid recomputation
  Field_Scalar alpha1_bar,
               H,
               H_lim,
               dH,
               dalpha1_bar;

  Field_Scalar alpha1_d,
               mod_grad_alpha1_bar,
               mod_grad_alpha1_d,
               Dt_alpha1_d,
               CV_alpha1_d,
               div_vel;

  Field_Vect normal,
             grad_alpha1_bar;

  Field_Vect grad_alpha1_d,
             vel;

  using gradient_type = decltype(samurai::make_gradient_order2<decltype(alpha1_bar)>());
  gradient_type gradient;

  using divergence_type = decltype(samurai::make_divergence_order2<decltype(normal)>());
  divergence_type divergence;

  double eps;                     // Tolerance when we want to avoid division by zero
  double mod_grad_alpha1_bar_min; // Minimum threshold for which not computing anymore the unit normal

  LinearizedBarotropicEOS<> EOS_phase1,
                            EOS_phase2; // The two variables which take care of the
                                        // barotropic EOS to compute the speed of sound

  samurai::RusanovFlux<Field> Rusanov_flux; // Auxiliary variable to compute the flux

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void update_geometry(); // Auxiliary routine to compute normals and curvature

  void init_variables(); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue

  void apply_relaxation(); // Apply the relaxation
};


// Implement class constructor
//
template<std::size_t dim>
TwoScaleCapillarity<dim>::TwoScaleCapillarity(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                                              const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                                              std::size_t min_level, std::size_t max_level,
                                              double Tf_, double cfl_, std::size_t nfiles_,
                                              bool apply_relax_, bool mass_transfer_):
  box(min_corner, max_corner), mesh(box, min_level, max_level, {false, true}),
  apply_relax(apply_relax_), Tf(Tf_), cfl(cfl_), mass_transfer(mass_transfer_), nfiles(nfiles_),
  gradient(samurai::make_gradient_order2<decltype(alpha1_bar)>()),
  divergence(samurai::make_divergence_order2<decltype(normal)>()),
  eps(1e-9), mod_grad_alpha1_bar_min(0.0),
  EOS_phase1(EquationData::p0_phase1, EquationData::rho0_phase1, EquationData::c0_phase1),
  EOS_phase2(EquationData::p0_phase2, EquationData::rho0_phase2, EquationData::c0_phase2),
  Rusanov_flux(EOS_phase1, EOS_phase2, eps, mod_grad_alpha1_bar_min) {
    std::cout << "Initializing variables " << std::endl;
    std::cout << std::endl;
    init_variables();
}


// Auxiliary routine to compute normals and curvature
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::update_geometry() {
  samurai::update_ghost_mr(alpha1_bar);

  grad_alpha1_bar = gradient(alpha1_bar);

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           mod_grad_alpha1_bar[cell] = std::sqrt(xt::sum(grad_alpha1_bar[cell]*grad_alpha1_bar[cell])());

                           if(mod_grad_alpha1_bar[cell] > mod_grad_alpha1_bar_min) {
                             normal[cell] = grad_alpha1_bar[cell]/mod_grad_alpha1_bar[cell];
                           }
                           else {
                             for(std::size_t d = 0; d < dim; ++d) {
                               normal[cell][d] = nan("");
                             }
                           }
                         });
  samurai::update_ghost_mr(normal);
  H = -divergence(normal);
}


// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::init_variables() {
  // Create conserved and auxiliary fields
  conserved_variables = samurai::make_field<double, EquationData::NVARS>("conserved", mesh);

  alpha1_bar          = samurai::make_field<double, 1>("alpha1_bar", mesh);
  grad_alpha1_bar     = samurai::make_field<double, dim>("grad_alpha1_bar", mesh);
  normal              = samurai::make_field<double, dim>("normal", mesh);
  H                   = samurai::make_field<double, 1>("H", mesh);
  H_lim               = samurai::make_field<double, 1>("Hlim", mesh);
  dH                  = samurai::make_field<double, 1>("dH", mesh);

  dalpha1_bar         = samurai::make_field<double, 1>("dalpha1_bar", mesh);

  mod_grad_alpha1_bar = samurai::make_field<double, 1>("mod_grad_alpha1_bar", mesh);

  alpha1_d            = samurai::make_field<double, 1>("alpha1_d", mesh);
  grad_alpha1_d       = samurai::make_field<double, dim>("grad_alpha1_d", mesh);
  mod_grad_alpha1_d   = samurai::make_field<double, 1>("mod_grad_alpha1_d", mesh);
  vel                 = samurai::make_field<double, dim>("vel", mesh);
  div_vel             = samurai::make_field<double, 1>("div_vel", mesh);
  Dt_alpha1_d         = samurai::make_field<double, 1>("Dt_alpha1_d", mesh);
  CV_alpha1_d         = samurai::make_field<double, 1>("CV_alpha1_d", mesh);

  // Declare some constant parameters associated to the grid and to the
  // initial state
  const double x0    = 1.0;
  const double y0    = 1.0;
  const double R     = 0.15;
  const double eps_R = 0.6*R;

  const double U_0 = 6.66;
  const double U_1 = 0.0;
  const double V   = 0.0;

  // Initialize some fields to define the bubble with a loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Set large-scale volume fraction
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];

                           const double r = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));

                           double w = (r >= R - 0.5*eps_R && r < R + 0.5*eps_R) ?
                                      std::max(0.5 + 0.5*std::tanh(-8.0*((r - R + eps / 2) / eps)), 0.0) :
                                      ((r < R - 0.5*eps_R) ? 1.0 : 0.0);
                           if(w < 1e-15) {
                             w = 0.0;
                           }

                           alpha1_bar[cell] = w;
                         });

  // Compute the geometrical quantities
  update_geometry();

  // Loop over a cell to complete the remaining variables
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Set small-scale variables
                           conserved_variables[cell][ALPHA1_D_INDEX] = 0.0;
                           alpha1_d[cell]                            = conserved_variables[cell][ALPHA1_D_INDEX];
                           conserved_variables[cell][SIGMA_D_INDEX]  = 0.0;
                           conserved_variables[cell][M1_D_INDEX]     = conserved_variables[cell][ALPHA1_D_INDEX]*EOS_phase1.get_rho0();

                           // Set mass large-scale phase 1
                           auto p1 = EOS_phase2.get_p0();
                           p1 += (alpha1_bar[cell] > 1.0 - eps) ? EquationData::sigma/R : ((alpha1_bar[cell] > 0.0) ? EquationData::sigma*H[cell] : 0.0);
                           const auto rho1 = EOS_phase1.rho_value(p1);

                           conserved_variables[cell][M1_INDEX] = (!std::isnan(rho1)) ?
                                                                 alpha1_bar[cell]*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX])*rho1 :
                                                                 0.0;

                           // Set mass phase 2
                           conserved_variables[cell][M2_INDEX] = (1.0 - alpha1_bar[cell])*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX])*
                                                                 EOS_phase2.get_rho0();

                           // Set conserved variable associated to large-scale volume fraction
                           const auto rho = conserved_variables[cell][M1_INDEX]
                                          + conserved_variables[cell][M2_INDEX]
                                          + conserved_variables[cell][M1_D_INDEX];

                           conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = rho*alpha1_bar[cell];

                           // Set momentum
                           conserved_variables[cell][RHO_U_INDEX]     = conserved_variables[cell][M1_INDEX]*U_1 + conserved_variables[cell][M2_INDEX]*U_0;
                           conserved_variables[cell][RHO_U_INDEX + 1] = rho*V;

                           vel[cell][0] = conserved_variables[cell][RHO_U_INDEX]/rho;
                           vel[cell][1] = conserved_variables[cell][RHO_U_INDEX + 1]/rho;
                         });

  // Set useful small-scale related fields
  grad_alpha1_d = gradient(alpha1_d);
  div_vel       = divergence(vel);
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           mod_grad_alpha1_d[cell] = std::sqrt(xt::sum(grad_alpha1_d[cell]*grad_alpha1_d[cell])());
                         });

  // Apply bcs
  const samurai::DirectionVector<dim> left = {-1, 0};
  const samurai::DirectionVector<dim> right = {1, 0};
  samurai::make_bc<samurai::Dirichlet<1>>(conserved_variables, 0.0, 1.0*EOS_phase2.get_rho0(), 0.0, 0.0, 0.0, 0.0, EOS_phase2.get_rho0()*U_0, 0.0)->on(left);
  samurai::make_bc<samurai::Neumann<1>>(conserved_variables, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)->on(right);
}


// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double TwoScaleCapillarity<dim>::get_max_lambda() const {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           // Compute the velocity along both horizontal and vertical direction
                           const auto rho   = conserved_variables[cell][M1_INDEX]
                                            + conserved_variables[cell][M2_INDEX]
                                            + conserved_variables[cell][M1_D_INDEX];
                           const auto vel_x = conserved_variables[cell][RHO_U_INDEX]/rho;
                           const auto vel_y = conserved_variables[cell][RHO_U_INDEX + 1]/rho;

                           // Compute frozen speed of sound
                           const auto alpha1    = alpha1_bar[cell]*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                           const auto rho1      = (alpha1 > eps) ? conserved_variables[cell][M1_INDEX]/alpha1 : nan("");
                           const auto alpha2    = 1.0 - alpha1 - conserved_variables[cell][ALPHA1_D_INDEX];
                           const auto rho2      = (alpha2 > eps) ? conserved_variables[cell][M2_INDEX]/alpha2 : nan("");
                           const auto c_squared = conserved_variables[cell][M1_INDEX]*EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                                + conserved_variables[cell][M2_INDEX]*EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);
                           const auto c         = std::sqrt(c_squared/rho)/(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);

                           // Add term due to surface tension
                           const double r = EquationData::sigma*std::sqrt(xt::sum(grad_alpha1_bar[cell]*grad_alpha1_bar[cell])())/(rho*c*c);

                           // Update eigenvalue estimate
                           res = std::max(std::max(std::abs(vel_x) + c*(1.0 + 0.125*r),
                                                   std::abs(vel_y) + c*(1.0 + 0.125*r)),
                                          res);
                         });

  return res;
}


// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void TwoScaleCapillarity<dim>::apply_relaxation() {
  // Apply relaxation with Newton method
  const double tol         = 1e-8; /*--- Tolerance of the Newton method ---*/
  const double lambda      = 0.9;  /*--- Parameter for bound preserving strategy ---*/
  const double alpha1d_max = 0.5;  /*--- Maximum allowed value of the small-scale volume fraction ---*/

  // Loop of Newton method
  std::size_t Newton_iter = 0;
  bool relaxation_applied = true;
  bool mass_transfer_NR   = mass_transfer; /*--- This value can change during the Newton loop, so we create a copy rather modyfing the original ---*/
  while(relaxation_applied == true) {
    relaxation_applied = false;
    Newton_iter++;

    // Perform the Newton step
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             // Reinitialization of partial masses in case of evanascent volume fraction
                             if(alpha1_bar[cell] < eps) {
                               conserved_variables[cell][M1_INDEX] = alpha1_bar[cell]*EOS_phase1.get_rho0();
                             }
                             if(1.0 - alpha1_bar[cell] < eps) {
                               conserved_variables[cell][M2_INDEX] = (1.0 - alpha1_bar[cell])*EOS_phase2.get_rho0();
                             }

                             const auto rho = conserved_variables[cell][M1_INDEX]
                                            + conserved_variables[cell][M2_INDEX]
                                            + conserved_variables[cell][M1_D_INDEX];

                             // Update auxiliary values affected by the nonlinear function for which we seek a zero
                             const auto alpha1 = alpha1_bar[cell]*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                             const auto rho1   = (alpha1 > eps) ? conserved_variables[cell][M1_INDEX]/alpha1 : nan("");
                             const auto p1     = EOS_phase1.pres_value(rho1);

                             const auto alpha2 = 1.0 - alpha1 - conserved_variables[cell][ALPHA1_D_INDEX];
                             const auto rho2   = (alpha2 > eps) ? conserved_variables[cell][M2_INDEX]/alpha2 : nan("");
                             const auto p2     = EOS_phase2.pres_value(rho2);

                             const auto rho1d  = (conserved_variables[cell][M1_D_INDEX] > eps && conserved_variables[cell][ALPHA1_D_INDEX] > eps) ?
                                                  conserved_variables[cell][M1_D_INDEX]/conserved_variables[cell][ALPHA1_D_INDEX] : EOS_phase1.get_rho0();

                             // Prepare for mass transfer if desired
                             if(mass_transfer_NR) {
                               if(3.0/(EquationData::kappa*rho1d)*rho1*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]) - (1.0 - alpha1_bar[cell]) > 0.0 &&
                                  alpha1_bar[cell] > 1e-2 && alpha1_bar[cell] < 1e-1 &&
                                  -grad_alpha1_bar[cell][0]*conserved_variables[cell][RHO_U_INDEX]
                                  -grad_alpha1_bar[cell][1]*conserved_variables[cell][RHO_U_INDEX + 1] > 0.0 &&
                                  conserved_variables[cell][ALPHA1_D_INDEX] < alpha1d_max) {
                                 H_lim[cell] = std::min(H[cell], EquationData::Hmax);
                               }
                               else {
                                 H_lim[cell] = H[cell];
                               }
                             }
                             else {
                               H_lim[cell] = H[cell];
                             }

                             dH[cell] = H[cell] - H_lim[cell]; //TODO: Initialize this outside and check if the maximum of dH
                                                               //at previous iteration is grater than a tolerance (1e-7 in Arthur's code).
                                                               //On the other hand, update geoemtry should in principle always be necessary,
                                                               //but seems to lead to issues if called every Newton iteration

                             // Compute the nonlinear function for which we seek the zero (basically the Laplace law)
                             const auto F = (1.0 - conserved_variables[cell][ALPHA1_D_INDEX])*(p1 - p2)
                                          - EquationData::sigma*H_lim[cell];

                             // Perform the relaxation only where really needed
                             if(!std::isnan(F) && std::abs(F) > tol*std::min(EOS_phase1.get_p0(), EquationData::sigma*H_lim[cell]) &&
                                std::abs(dalpha1_bar[cell]) > tol && alpha1_bar[cell] > eps && 1.0 - alpha1_bar[cell] > eps) {
                               relaxation_applied = true;

                               // Compute the derivative w.r.t large scale volume fraction recalling that for a barotropic EOS dp/drho = c^2
                               const auto dF_dalpha1_bar = -conserved_variables[cell][M1_INDEX]/(alpha1_bar[cell]*alpha1_bar[cell])*
                                                            EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)
                                                           -conserved_variables[cell][M2_INDEX]/((1.0 - alpha1_bar[cell])*(1.0 - alpha1_bar[cell]))*
                                                            EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2);

                               // Compute the psuedo time step starting as initial guess from the ideal unmodified Newton method
                               double dtau_ov_epsilon = std::numeric_limits<double>::infinity();

                               // Bound preserving condition for m1, velocity and small-scale volume fraction
                               if(dH[cell] > 0.0 && !std::isnan(rho1)) {
                                 /*--- Bound preserving condition for m1 ---*/
                                 dtau_ov_epsilon = lambda*conserved_variables[cell][M1_INDEX]*(1.0 - alpha1_bar[cell])/
                                                   (rho1*EquationData::sigma*dH[cell]);
                                 if(dtau_ov_epsilon < 0.0) {
                                   std::cerr << "Negative time step found after relaxation of mass of large-scale phase 1" << std::endl;
                                   exit(1);
                                 }

                                 /*--- Bound preserving for the velocity ---*/
                                 const auto mom_dot_vel = (conserved_variables[cell][RHO_U_INDEX]*conserved_variables[cell][RHO_U_INDEX] +
                                                           conserved_variables[cell][RHO_U_INDEX + 1]*conserved_variables[cell][RHO_U_INDEX + 1])/rho;
                                 const auto fac = std::max(3.0/(EquationData::kappa*rho1d)*(rho1/(1.0 - alpha1_bar[cell])) -
                                                           1.0/(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]), 0.0);
                                 if(fac > 0.0) {
                                   auto dtau_ov_epsilon_tmp = mom_dot_vel/(EquationData::Hmax*dH[cell]*fac*EquationData::sigma*EquationData::sigma);
                                   dtau_ov_epsilon          = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                   if(dtau_ov_epsilon < 0.0) {
                                     std::cerr << "Negative time step found after relaxation of velocity" << std::endl;
                                     exit(1);
                                   }
                                 }

                                 /*--- Bound preserving for the small-scale volume fraction ---*/
                                 auto dtau_ov_epsilon_tmp = lambda*(alpha1d_max - conserved_variables[cell][ALPHA1_D_INDEX])*(1.0 - alpha1_bar[cell])*rho1d/
                                                            (rho1*EquationData::sigma*dH[cell]);
                                 dtau_ov_epsilon          = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                 if(conserved_variables[cell][ALPHA1_D_INDEX] > 0.0 && conserved_variables[cell][ALPHA1_D_INDEX] < alpha1d_max) {
                                   dtau_ov_epsilon_tmp = conserved_variables[cell][ALPHA1_D_INDEX]*(1.0 - alpha1_bar[cell])*rho1d/
                                                         (rho1*EquationData::sigma*dH[cell]);

                                   dtau_ov_epsilon     = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                                 }
                                 if(dtau_ov_epsilon < 0.0) {
                                   std::cerr << "Negative time step found after relaxation of small-scale volume fraction" << std::endl;
                                   exit(1);
                                 }
                               }

                               // Bound preserving condition for large-scale volume fraction
                               const auto dF_dalpha1d   = p2 - p1
                                                        + EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)*rho1
                                                        - EOS_phase2.c_value(rho2)*EOS_phase2.c_value(rho2)*rho2;
                               const auto dF_dm1        = EOS_phase1.c_value(rho1)*EOS_phase1.c_value(rho1)/alpha1_bar[cell];
                               const auto R             = dF_dalpha1d/rho1d - dF_dm1;
                               const auto a             = rho1*EquationData::sigma*dH[cell]*R/
                                                          ((1.0 - alpha1_bar[cell])*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]));
                               /*--- Upper bound ---*/
                               auto b                   = (F + lambda*(1.0 - alpha1_bar[cell])*dF_dalpha1_bar)/
                                                          (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                               auto D                   = b*b - 4.0*a*(-lambda*(1.0 - alpha1_bar[cell]));
                               auto dtau_ov_epsilon_tmp = std::numeric_limits<double>::infinity();
                               if(D > 0.0 && (a > 0.0 || (a < 0.0 && b > 0.0))) {
                                 dtau_ov_epsilon_tmp = 0.5*(-b + std::sqrt(D))/a;
                               }
                               if(a == 0.0 && b > 0.0) {
                                 dtau_ov_epsilon_tmp = lambda*(1.0 - alpha1_bar[cell])/b;
                               }
                               dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                               /*--- Lower bound ---*/
                               dtau_ov_epsilon_tmp = std::numeric_limits<double>::infinity();
                               b                   = (F - lambda*alpha1_bar[cell]*dF_dalpha1_bar)/
                                                     (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                               D                   = b*b - 4.0*a*(lambda*alpha1_bar[cell]);
                               if(D > 0.0 && (a < 0.0 || (a > 0.0 && b < 0.0))) {
                                 dtau_ov_epsilon_tmp = 0.5*(-b - std::sqrt(D))/a;
                               }
                               if(a == 0.0 && b < 0.0) {
                                 dtau_ov_epsilon_tmp = -lambda*alpha1_bar[cell]/b;
                               }
                               dtau_ov_epsilon = std::min(dtau_ov_epsilon, dtau_ov_epsilon_tmp);
                               if(dtau_ov_epsilon < 0.0) {
                                 std::cerr << "Negative time step found after relaxation of large-scale volume fraction" << std::endl;
                                 exit(1);
                               }

                               // Compute the effective variation of the variables
                               if(std::isinf(dtau_ov_epsilon)) {
                                 // If we are in this branch we do not have mass transfer
                                 // and we do not have other restrictions on the bounds of large scale volume fraction
                                 dalpha1_bar[cell] = -F/dF_dalpha1_bar;
                               }
                               else {
                                 const auto dm1 = -dtau_ov_epsilon/(1.0 - alpha1_bar[cell])*
                                                   (conserved_variables[cell][M1_INDEX]/(alpha1_bar[cell]*(1.0 - conserved_variables[cell][ALPHA1_D_INDEX])))*
                                                   EquationData::sigma*dH[cell];

                                 const auto num_dalpha1_bar = dtau_ov_epsilon/(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                                 const auto den_dalpha1_bar = 1.0 - dtau_ov_epsilon*dF_dalpha1_bar/(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                                 dalpha1_bar[cell]          = (num_dalpha1_bar/den_dalpha1_bar)*(F - dm1*(dF_dm1 + dtau_ov_epsilon*(dF_dalpha1d/rho1d)));

                                 if(dm1 > 0.0) {
                                   std::cerr << "Negative sign of mass transfer inside Newton step" << std::endl;
                                   exit(1);
                                 }
                                 else {
                                   conserved_variables[cell][M1_INDEX] += dm1;
                                   if(conserved_variables[cell][M1_INDEX] < 0.0) {
                                     std::cerr << "Negative mass of large-scale phase 1 inside Newton step" << std::endl;
                                   }
                                   conserved_variables[cell][M1_D_INDEX] -= dm1;
                                   if(conserved_variables[cell][M1_D_INDEX] < 0.0) {
                                     std::cerr << "Negative mass of small-scale phase 1 inside Newton step" << std::endl;
                                   }
                                 }

                                 if(conserved_variables[cell][ALPHA1_D_INDEX] - dm1/rho1d > 1.0) {
                                   std::cerr << "Exceeding value for small-scale volume fraction inside Newton step " << std::endl;
                                   exit(1);
                                 }
                                 else {
                                   conserved_variables[cell][ALPHA1_D_INDEX] -= dm1/rho1d;
                                 }

                                 conserved_variables[cell][SIGMA_D_INDEX] -= dm1*3.0*EquationData::Hmax/(EquationData::kappa*rho1d);
                               }

                               if(alpha1_bar[cell] + dalpha1_bar[cell] < 0.0 && alpha1_bar[cell] + dalpha1_bar[cell] > 1.0) {
                                 std::cerr << "Bounds exceeding value for large-scale volume fraction inside Newton step " << std::endl;
                               }
                               else {
                                 alpha1_bar[cell] += dalpha1_bar[cell];
                               }

                               if(dH[cell] > 0.0) {
                                 const auto fac = std::max(3.0/(EquationData::kappa*rho1d)*(rho1/(1.0 - alpha1_bar[cell])) -
                                                           1.0/(1.0 - conserved_variables[cell][ALPHA1_D_INDEX]), 0.0);

                                 double drho_fac = 0.0;
                                 const auto mom_squared = conserved_variables[cell][RHO_U_INDEX]*conserved_variables[cell][RHO_U_INDEX]
                                                        + conserved_variables[cell][RHO_U_INDEX + 1]*conserved_variables[cell][RHO_U_INDEX + 1];
                                 if(mom_squared > 0.0) {
                                    drho_fac = dtau_ov_epsilon*
                                               EquationData::sigma*EquationData::sigma*dH[cell]*fac*H_lim[cell]*rho/mom_squared;
                                 }

                                 for(std::size_t d = 0; d < EquationData::dim; ++d) {
                                   conserved_variables[cell][RHO_U_INDEX + d] -= drho_fac*conserved_variables[cell][RHO_U_INDEX + d];
                                 }
                               }
                             }

                             // Update "conservative counter part" of large-scale volume fraction.
                             // Do it outside because this can change either because of relaxation of
                             // alpha1_bar or because of change of rho for evanescent volume fractions.
                             conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = rho*alpha1_bar[cell];
                           });

    // Recompute geometric quantities (curvature potentially changed in the Newton loop)
    //update_geometry();

    // Stop the mass transfer after a sufficient time of Newton iterations for safety
    if(mass_transfer_NR && Newton_iter > 30) {
      mass_transfer_NR = false;
    }

    // Newton cycle diverged
    if(Newton_iter > 60) {
      std::cout << "Netwon method not converged" << std::endl;
      save(fs::current_path(), "FV_two_scale_capillarity", "_diverged",
           conserved_variables, alpha1_bar, grad_alpha1_bar, normal, H, H_lim, mod_grad_alpha1_bar);
      exit(1);
    }
  }
}


// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void TwoScaleCapillarity<dim>::save(const fs::path& path,
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
template<std::size_t dim>
void TwoScaleCapillarity<dim>::run() {
  // Default output arguemnts
  fs::path path        = fs::current_path();
  std::string filename = "FV_two_scale_capillarity";
  const double dt_save = Tf / static_cast<double>(nfiles);

  // Auxiliary variables to save updated fields
  auto conserved_variables_np1 = samurai::make_field<double, EquationData::NVARS>("conserved_np1", mesh);

  // Create the flux variable
  auto numerical_flux = Rusanov_flux.make_two_scale_capillarity(grad_alpha1_bar);

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, conserved_variables, alpha1_bar, grad_alpha1_bar, normal, H, mod_grad_alpha1_bar,
                                    grad_alpha1_d, mod_grad_alpha1_d, vel, div_vel);

  // Start the loop
  std::size_t nsave = 0;
  std::size_t nt    = 0;
  double t          = 0.0;
  double dx         = samurai::cell_length(mesh[mesh_id_t::cells].max_level());
  double dt         = std::min(Tf - t, cfl*dx/get_max_lambda());
  while(t != Tf) {
    t += dt;

    std::cout << fmt::format("Iteration {}: t = {}, dt = {}", ++nt, t, dt) << std::endl;

    // Apply mesh adaptation
    samurai::update_ghost_mr(grad_alpha1_bar);
    auto MRadaptation = samurai::make_MRAdapt(grad_alpha1_bar);
    MRadaptation(1e-5, 0, conserved_variables, alpha1_bar);
    // Resize the fields to be recomputed
    normal.resize();
    H.resize();
    H_lim.resize();
    mod_grad_alpha1_bar.resize();
    update_geometry();

    // Apply the numerical scheme without relaxation
    samurai::update_ghost_mr(conserved_variables);
    samurai::update_bc(conserved_variables);
    auto flux_conserved = numerical_flux(conserved_variables);
    conserved_variables_np1.resize();
    conserved_variables_np1 = conserved_variables - dt*flux_conserved;

    std::swap(conserved_variables.array(), conserved_variables_np1.array());

    // Sanity check (and numerical artefacts to clear data) after hyperbolic step
    // and before Newton loop (if desired).
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             // Sanity check for rho_alpha1_bar
                             if(conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] < 0.0) {
                               if(conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] < -1e-10) {
                                 std::cerr << " Negative large-scale mass phase 1 at the beginning of the relaxation" << std::endl;
                                 exit(1);
                               }
                               conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = 0.0;
                             }
                             // Sanity check for m1
                             if(conserved_variables[cell][M1_INDEX] < 0.0) {
                               if(conserved_variables[cell][M1_INDEX] < -1e-14) {
                                 std::cerr << "Negative mass for phase 1" << std::endl;
                                 exit(1);
                               }
                               conserved_variables[cell][M1_INDEX] = 0.0;
                             }
                             // Sanity check for m2
                             if(conserved_variables[cell][M2_INDEX] < 0.0) {
                               if(conserved_variables[cell][M2_INDEX] < -1e-14) {
                                 std::cerr << "Negative mass for phase 2" << std::endl;
                                 exit(1);
                               }
                               conserved_variables[cell][M2_INDEX] = 0.0;
                             }
                             // Sanity check for alpha1_d
                             if(conserved_variables[cell][ALPHA1_D_INDEX] > 1.0) {
                               std::cerr << "Exceding value for small-scale volume fraction" << std::endl;
                               exit(1);
                             }

                             const auto rho = conserved_variables[cell][M1_INDEX]
                                            + conserved_variables[cell][M2_INDEX]
                                            + conserved_variables[cell][M1_D_INDEX];

                             alpha1_bar[cell] = std::min(std::max(conserved_variables[cell][RHO_ALPHA1_BAR_INDEX]/rho, 0.0), 1.0);
                           });
    update_geometry();

    if(apply_relax) {
      // Apply relaxation if desired, which will modify alpha1_bar and, consequently, for what
      // concerns next time step, rho_alpha1_bar
      // (as well as grad_alpha1_bar, updated dynamically in Newton since curvature potentially changes),
      // In the case of mass transfer, also other state varaibles are modified.
      dalpha1_bar.resize();
      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               dalpha1_bar[cell] = std::numeric_limits<double>::infinity();
                             });
      apply_relaxation();
      update_geometry();
    }

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      // Compute small-scale geometric related quantities
      alpha1_d.resize();
      grad_alpha1_d.resize();
      mod_grad_alpha1_d.resize();
      vel.resize();
      div_vel.resize();
      Dt_alpha1_d.resize();
      CV_alpha1_d.resize();

      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               alpha1_d[cell] = conserved_variables[cell][ALPHA1_D_INDEX];

                               const auto rho = conserved_variables[cell][M1_INDEX]
                                              + conserved_variables[cell][M2_INDEX]
                                              + conserved_variables[cell][M1_D_INDEX];
                               vel[cell][0]   = conserved_variables[cell][RHO_U_INDEX]/rho;
                               vel[cell][1]   = conserved_variables[cell][RHO_U_INDEX + 1]/rho;
                             });

      grad_alpha1_d = gradient(alpha1_d);
      div_vel       = divergence(vel);

      samurai::for_each_cell(mesh,
                             [&](const auto& cell)
                             {
                               mod_grad_alpha1_d[cell] = std::sqrt(xt::sum(grad_alpha1_d[cell]*grad_alpha1_d[cell])());

                               Dt_alpha1_d[cell] = (conserved_variables[cell][ALPHA1_D_INDEX] - conserved_variables_np1[cell][ALPHA1_D_INDEX])/dt
                                                 + vel[cell][0]*grad_alpha1_d[cell][0] + vel[cell][1]*grad_alpha1_d[cell][1];

                               CV_alpha1_d[cell] = Dt_alpha1_d[cell] + conserved_variables[cell][ALPHA1_D_INDEX]*div_vel[cell];
                             });

      // Perform saving
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix, conserved_variables, alpha1_bar, grad_alpha1_bar, normal, H, H_lim, mod_grad_alpha1_bar,
                                   grad_alpha1_d, mod_grad_alpha1_d, vel, div_vel, Dt_alpha1_d, CV_alpha1_d);
    }

    // Compute updated time step
    dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());
    dt = std::min(Tf - t, cfl*dx/get_max_lambda());
  }
}
