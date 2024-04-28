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

#include "two_scale_FV.hpp"

#include <samurai/mr/adapt.hpp>

// Specify the use of this namespace where we just store the indices
// and, in this case, some parameters related to EOS
using namespace EquationData;

// This is the class for the simulation of a two-scale model
//
template<std::size_t dim>
class TwoScale {
public:
  using Config = samurai::MRConfig<dim>;

  TwoScale() = default; // Default constructor. This will do nothing
                        // and basically will never be used

  TwoScale(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
           const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
           std::size_t min_level, std::size_t max_level,
           double Tf_, double cfl_, std::size_t nfiles_ = 100,
           bool apply_relax_ = true);  // Class constrcutor with the arguments related
                                      // to the grid, to the physics and to the relaxation.
                                      // Maybe in the future,
                                      // we could think to add parameters related to EOS

  bool check_apply_relaxation() const; // Check whether we applied relaxation or not

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

  LinearizedBarotropicEOS EOS_phase1,
                          EOS_phase2; // The two varaibles which take care of the
                                      // barotropic EOS to compute the speed of sound

  bool apply_relax; // Choose whether to apply or not the relaxation

  double Tf;  // Final time of the simulation
  double cfl; // Courant number of the simulation so as to compute the time step

  std::size_t nfiles; // Number of files desired for output

  Field conserved_variables; // The variable which stores the conserved variables,
                             // namely the varialbes for which we solve a PDE system

  // Now we declare a bunch of fields which depend from the state, but it is useful
  // to have it so as to avoid recomputation
  Field_Scalar rho,
               alpha1_bar,
               alpha2_bar,
               alpha1,
               rho1,
               alpha2,
               rho2,
               p_bar,
               c,
               alpha1_d;

  Field_Vect vel;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void init_variables(); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  void impose_left_dirichet_BC(); // Impose Dirichlet boundary conditions for left boundary (this is problem dependent)

  void impose_right_dirichet_BC(); // Impose Dirichlet boundary conditions for right boundary (this is problem dependent)

  double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue

  void apply_relaxation_linearized_EOS(); // Apply the relaxation specific for linearized EOS

  void apply_relaxation(); // Apply the relaxation

  void update_auxiliary_fields_pre_relaxation(); // Update auxiliary fields which are not touched by relaxation

  void update_auxiliary_fields_post_relaxation(); // Update auxiliary fields after relaxation are not touched by relaxation
};


// Implement class constructor
//
template<std::size_t dim>
TwoScale<dim>::TwoScale(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                        const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                        std::size_t min_level, std::size_t max_level,
                        double Tf_, double cfl_, std::size_t nfiles_,
                        bool apply_relax_):
  box(min_corner, max_corner), mesh(box, min_level, max_level, {false, false}),
  apply_relax(apply_relax_), Tf(Tf_), cfl(cfl_), nfiles(nfiles_) {
    EOS_phase1 = LinearizedBarotropicEOS(p0_phase1, rho0_phase1, c0_phase1);
    EOS_phase2 = LinearizedBarotropicEOS(p0_phase2, rho0_phase2, c0_phase2);

    init_variables();
    impose_left_dirichet_BC();
    impose_right_dirichet_BC();

    /*--- Impose Neumann bcs on the top and bottom boundaries ---*/
    samurai::DirectionVector<dim> top    = {0, 1};
    samurai::DirectionVector<dim> bottom = {0, -1};
    samurai::make_bc<samurai::Neumann>(conserved_variables, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)->on(top);
    samurai::make_bc<samurai::Neumann>(conserved_variables, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)->on(bottom);
    samurai::make_bc<samurai::Neumann>(vel, 0.0, 0.0)->on(top);
    samurai::make_bc<samurai::Neumann>(vel, 0.0, 0.0)->on(bottom);
    samurai::make_bc<samurai::Neumann>(p_bar, 0.0)->on(top);
    samurai::make_bc<samurai::Neumann>(p_bar, 0.0)->on(bottom);
    samurai::make_bc<samurai::Neumann>(c, 0.0)->on(top);
    samurai::make_bc<samurai::Neumann>(c, 0.0)->on(bottom);
}


// Auxiliary routine to check whether we applied relaxation or not
//
template<std::size_t dim>
bool TwoScale<dim>::check_apply_relaxation() const {
  return apply_relax;
}


// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void TwoScale<dim>::init_variables() {
  // Create conserved and auxiliary fields
  conserved_variables = samurai::make_field<double, EquationData::NVARS>("conserved", mesh);

  rho        = samurai::make_field<double, 1>("rho", mesh);
  vel        = samurai::make_field<double, dim>("vel", mesh);
  alpha1_bar = samurai::make_field<double, 1>("alpha1_bar", mesh);
  alpha2_bar = samurai::make_field<double, 1>("alpha2_bar", mesh);
  alpha1     = samurai::make_field<double, 1>("alpha1", mesh);
  rho1       = samurai::make_field<double, 1>("rho1", mesh);
  alpha2     = samurai::make_field<double, 1>("alpha2", mesh);
  rho2       = samurai::make_field<double, 1>("rho2", mesh);
  p_bar      = samurai::make_field<double, 1>("p_bar", mesh);
  c          = samurai::make_field<double, 1>("c", mesh);
  alpha1_d   = samurai::make_field<double, 1>("alpha1_d", mesh);

  // Declare some constant parameters associated to the grid and to the
  // initial state
  const double x0 = 1.0;
  const double y0 = 0.5;

  const double xd = 0.3;

  const double eps = 1e-7;

  // Initialize the fields with a loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];
                           const double y    = center[1];

                           if(x <= xd) {
                             const double alpha1_bar_0 = 1.0 - eps;
                             const double alpha1_d_0   = 0.0;
                             const double rho1_0       = 100.0;
                             const double rho2_0       = 1e4;
                             const double u_0          = 0.0;
                             const double v_0          = 0.0;

                             conserved_variables[cell][M1_INDEX]       = alpha1_bar_0*rho1_0*(1.0 - alpha1_d_0);
                             conserved_variables[cell][M2_INDEX]       = (1.0 - alpha1_bar_0)*rho2_0*(1.0 - alpha1_d_0);
                             conserved_variables[cell][M1_D_INDEX]     = rho0_phase1*alpha1_d_0;
                             conserved_variables[cell][ALPHA1_D_INDEX] = alpha1_d_0;

                             rho[cell] = conserved_variables[cell][M1_INDEX]
                                       + conserved_variables[cell][M2_INDEX]
                                       + conserved_variables[cell][M1_D_INDEX];

                             conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = rho[cell]*alpha1_bar_0;
                             conserved_variables[cell][RHO_U_INDEX]          = rho[cell]*u_0;
                             conserved_variables[cell][RHO_V_INDEX]          = rho[cell]*v_0;

                             vel[cell][0] = u_0;
                             vel[cell][1] = v_0;

                             alpha1_bar[cell] = alpha1_bar_0;

                             rho1[cell] = rho1_0;

                             rho2[cell] = rho2_0;
                           }
                           else {
                             const double alpha1_bar_0 = eps;
                             const double alpha1_d_0   = 0.0;
                             const double rho1_0       = 1.0;
                             const double rho2_0       = 1e3;
                             const double u_0          = 0.0;
                             const double v_0          = 0.0;

                             conserved_variables[cell][M1_INDEX]       = alpha1_bar_0*rho1_0*(1.0 - alpha1_d_0);
                             conserved_variables[cell][M2_INDEX]       = (1.0 - alpha1_bar_0)*rho2_0*(1.0 - alpha1_d_0);
                             conserved_variables[cell][M1_D_INDEX]     = rho0_phase1*alpha1_d_0;
                             conserved_variables[cell][ALPHA1_D_INDEX] = alpha1_d_0;

                             rho[cell] = conserved_variables[cell][M1_INDEX]
                                       + conserved_variables[cell][M2_INDEX]
                                       + conserved_variables[cell][M1_D_INDEX];

                             conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = rho[cell]*alpha1_bar_0;
                             conserved_variables[cell][RHO_U_INDEX]          = rho[cell]*u_0;
                             conserved_variables[cell][RHO_V_INDEX]          = rho[cell]*v_0;

                             vel[cell][0] = u_0;
                             vel[cell][1] = v_0;

                             alpha1_bar[cell] = alpha1_bar_0;

                             rho1[cell] = rho1_0;

                             rho2[cell] = rho2_0;
                           }

                           // Identify the beam
                           if(std::abs(y - y0) < 0.1 && std::abs(x - x0) < 0.5) {
                             const double alpha1_bar_0 = eps;
                             const double alpha1_d_0   = 0.4;
                             const double rho1_0       = 1.0;
                             const double rho2_0       = 1e3;
                             const double u_0          = 0.0;
                             const double v_0          = 0.0;

                             conserved_variables[cell][M1_INDEX]       = alpha1_bar_0*rho1_0*(1.0 - alpha1_d_0);
                             conserved_variables[cell][M2_INDEX]       = (1.0 - alpha1_bar_0)*rho2_0*(1.0 - alpha1_d_0);
                             conserved_variables[cell][M1_D_INDEX]     = rho0_phase1*alpha1_d_0;
                             conserved_variables[cell][ALPHA1_D_INDEX] = alpha1_d_0;

                             rho[cell] = conserved_variables[cell][M1_INDEX]
                                       + conserved_variables[cell][M2_INDEX]
                                       + conserved_variables[cell][M1_D_INDEX];

                             conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = rho[cell]*alpha1_bar_0;
                             conserved_variables[cell][RHO_U_INDEX]          = rho[cell]*u_0;
                             conserved_variables[cell][RHO_V_INDEX]          = rho[cell]*v_0;

                             vel[cell][0] = u_0;
                             vel[cell][1] = v_0;

                             alpha1_bar[cell] = alpha1_bar_0;

                             rho1[cell] = rho1_0;

                             rho2[cell] = rho2_0;
                           }

                           alpha2_bar[cell] = 1.0 - alpha1_bar[cell];

                           alpha1[cell] = alpha1_bar[cell]*
                                          (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);

                           alpha2[cell] = 1.0
                                        - alpha1[cell]
                                        - conserved_variables[cell][ALPHA1_D_INDEX];

                           p_bar[cell] = alpha1_bar[cell]*EOS_phase1.pres_value(rho1[cell])
                                       + alpha2_bar[cell]*EOS_phase2.pres_value(rho2[cell]);

                           const double c_squared = conserved_variables[cell][M1_INDEX]*EOS_phase1.c_value(rho1[cell])*EOS_phase1.c_value(rho1[cell])
                                                  + conserved_variables[cell][M2_INDEX]*EOS_phase2.c_value(rho2[cell])*EOS_phase2.c_value(rho2[cell]);
                           c[cell] = std::sqrt(c_squared/rho[cell])/
                                     (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);

                           alpha1_d[cell] = conserved_variables[cell][ALPHA1_D_INDEX];
                         });
}


// Imposition of Dirichlet boundary condition for left boundary
//
template<std::size_t dim>
void TwoScale<dim>::impose_left_dirichet_BC() {
  // Create the state
  const double eps = 1e-7;

  const double alpha1_bar_L = 1.0 - eps;
  const double alpha1_d_L   = 0.0;
  const double rho1_L       = 100.0;
  const double rho2_L       = 1e4;
  const double u_L          = 0.0;
  const double v_L          = 0.0;

  const double m1_L   = alpha1_bar_L*rho1_L*(1.0 - alpha1_d_L);
  const double m2_L   = (1.0 - alpha1_bar_L)*rho2_L*(1.0 - alpha1_d_L);
  const double m1_d_L = rho0_phase1*alpha1_d_L;

  const double rho_L = m1_L + m2_L + m1_d_L;

  const double rho_alpha1_bar_L = rho_L*alpha1_bar_L;
  const double rho_u_L          = rho_L*u_L;
  const double rho_v_L          = rho_L*v_L;

  // Impose BC for the conserved field
  samurai::DirectionVector<dim> left = {-1, 0};

  samurai::make_bc<samurai::Dirichlet>(conserved_variables, m1_L, m2_L, m1_d_L, alpha1_d_L, rho_alpha1_bar_L, rho_u_L, rho_v_L)->on(left);

  // Impose BC for the velocity(min_corner, max_corner)
  samurai::make_bc<samurai::Dirichlet>(vel, u_L, v_L)->on(left);

  // Impose BC for the pressure
  const double p_bar_L = alpha1_bar_L*EOS_phase1.pres_value(rho1_L) + (1.0 - alpha1_bar_L)*EOS_phase2.pres_value(rho2_L);

  samurai::make_bc<samurai::Dirichlet>(p_bar, p_bar_L)->on(left);

  // Impose BC for the speed of sound
  const double c_squared = m1_L*EOS_phase1.c_value(rho1_L)*EOS_phase1.c_value(rho1_L)
                         + m2_L*EOS_phase2.c_value(rho2_L)*EOS_phase2.c_value(rho2_L);
  const double c_L       = std::sqrt(c_squared/rho_L)/(1.0 - alpha1_d_L);

  samurai::make_bc<samurai::Dirichlet>(c, c_L)->on(left);
}


// Imposition of Dirichlet boundary condition for right boundary
//
template<std::size_t dim>
void TwoScale<dim>::impose_right_dirichet_BC() {
  /// Create the state
  const double eps = 1e-7;

  const double alpha1_bar_R = eps;
  const double alpha1_d_R   = 0.0;
  const double rho1_R       = 1.0;
  const double rho2_R       = 1e3;
  const double u_R          = 0.0;
  const double v_R          = 0.0;

  const double m1_R   = alpha1_bar_R*rho1_R*(1.0 - alpha1_d_R);
  const double m2_R   = (1.0 - alpha1_bar_R)*rho2_R*(1.0 - alpha1_d_R);
  const double m1_d_R = rho0_phase1*alpha1_d_R;

  const double rho_R = m1_R + m2_R + m1_d_R;

  const double rho_alpha1_bar_R = rho_R*alpha1_bar_R;
  const double rho_u_R          = rho_R*u_R;
  const double rho_v_R          = rho_R*v_R;

  // Impose BC for the conserved field
  samurai::DirectionVector<dim> right = {1, 0};

  samurai::make_bc<samurai::Dirichlet>(conserved_variables, m1_R, m2_R, m1_d_R, alpha1_d_R, rho_alpha1_bar_R, rho_u_R, rho_v_R)->on(right);

  // Impose BC for the velocity
  samurai::make_bc<samurai::Dirichlet>(vel, u_R, v_R)->on(right);

  // Impose BC for the pressure
  const double p_bar_R = alpha1_bar_R*EOS_phase1.pres_value(rho1_R) + (1.0 - alpha1_bar_R)*EOS_phase2.pres_value(rho2_R);

  samurai::make_bc<samurai::Dirichlet>(p_bar, p_bar_R)->on(right);

  // Impose BC for the speed of sound
  const double c_squared = m1_R*EOS_phase1.c_value(rho1_R)*EOS_phase1.c_value(rho1_R)
                         + m2_R*EOS_phase2.c_value(rho2_R)*EOS_phase2.c_value(rho2_R);
  const double c_R       = std::sqrt(c_squared/rho_R)/(1.0 - alpha1_d_R);

  samurai::make_bc<samurai::Dirichlet>(c, c_R)->on(right);
}


// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double TwoScale<dim>::get_max_lambda() const {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           res = std::max(std::max(std::abs(vel[cell][0]) + c[cell],
                                                   std::abs(vel[cell][1]) + c[cell]),
                                          res);
                         });

  return res;
}


// Apply the relaxation specific for the linearized EOS
//
template<std::size_t dim>
void TwoScale<dim>::apply_relaxation_linearized_EOS() {
  auto alpha1_bar_rho1 = samurai::make_field<double, 1>("alpha1_bar_rho1", mesh);
  auto alpha2_bar_rho2 = samurai::make_field<double, 1>("alpha2_bar_rho2", mesh);
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           alpha1_bar_rho1[cell] = conserved_variables[cell][M1_INDEX]/
                                                   (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);

                           alpha2_bar_rho2[cell] = conserved_variables[cell][M2_INDEX]/
                                                   (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);
                         });

  const auto q      = EOS_phase2.get_rho0()*EOS_phase2.get_c0()*EOS_phase2.get_c0()
                    - EOS_phase1.get_rho0()*EOS_phase1.get_c0()*EOS_phase1.get_c0();
  const auto qtilde = alpha2_bar_rho2*EOS_phase2.get_c0()*EOS_phase2.get_c0();
                    - alpha1_bar_rho1*EOS_phase1.get_c0()*EOS_phase1.get_c0();

  const auto betaPos = (q - qtilde +
                        xt::sqrt((q - qtilde)*(q - qtilde) +
                                 4.0*alpha1_bar_rho1*EOS_phase1.get_c0()*EOS_phase1.get_c0()*
                                     alpha2_bar_rho2*EOS_phase2.get_c0()*EOS_phase2.get_c0()))/
                        (2.0*alpha2_bar_rho2*EOS_phase2.get_c0()*EOS_phase2.get_c0());

  alpha1_bar         = betaPos/(1.0 + betaPos);
}


// Apply the relaxation. This procedure is valid for a generic EOS
//
template<std::size_t dim>
void TwoScale<dim>::apply_relaxation() {
  // Apply relaxation with Newton method
  const double eps    = 1e-9; /*--- Tolerance of pure phase ---*/
  const double tol    = 1e-3; /*--- Tolerance of the Newton method ---*/
  const double lambda = 0.9;  /*--- Parameter for bound preserving strategy ---*/


  std::size_t Newton_iter = 0;
  bool relaxation_applied = true;
  while(relaxation_applied == true) {
    Newton_iter++;
    relaxation_applied = false;
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             // Compute partial densities since alpha1_bar is potentially changed
                             rho1[cell] = conserved_variables[cell][M1_INDEX]/
                                          (alpha1_bar[cell]*
                                           (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]));

                             rho2[cell] = conserved_variables[cell][M2_INDEX]/
                                          ((1.0 - alpha1_bar[cell])*
                                           (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]));

                             const double F = (1.0 - conserved_variables[cell][ALPHA1_D_INDEX])*
                                              (EOS_phase1.pres_value(rho1[cell]) - EOS_phase2.pres_value(rho2[cell]));

                             if(std::abs(F) > tol*EOS_phase1.get_p0() && alpha1_bar[cell] > eps && 1.0 - alpha1_bar[cell] > eps) {
                               relaxation_applied = true;

                               // Compute the derivative recalling that for a barotropic EOS dp/drho = c^2
                               const double dF = -conserved_variables[cell][M1_INDEX]/(alpha1_bar[cell]*alpha1_bar[cell])*
                                                  EOS_phase1.c_value(rho1[cell])*EOS_phase1.c_value(rho1[cell])
                                                 -conserved_variables[cell][M2_INDEX]/((1.0 - alpha1_bar[cell])*(1.0 - alpha1_bar[cell]))*
                                                  EOS_phase2.c_value(rho2[cell])*EOS_phase2.c_value(rho2[cell]);

                               // Apply Newton method
                               const double dalpha1_bar = -F/dF;
                               alpha1_bar[cell] += dalpha1_bar < 0 ? std::max(dalpha1_bar, -lambda*alpha1_bar[cell]) :
                                                                     std::min(dalpha1_bar, lambda*(1.0 - alpha1_bar[cell]));
                             }
                           });
    if(Newton_iter > 50) {
      std::cout << "Netwon method not converged" << std::endl;
      save(fs::current_path(), "FV_two_scale", "_diverged", conserved_variables, vel, alpha1_bar, alpha1, p_bar, c, rho1, rho2);
      exit(1);
    }
  }
}


// Update auxiliary fields which are not modified by the relaxation
//
template<std::size_t dim>
void TwoScale<dim>::update_auxiliary_fields_pre_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           rho[cell] = conserved_variables[cell][M1_INDEX]
                                     + conserved_variables[cell][M2_INDEX]
                                     + conserved_variables[cell][M1_D_INDEX];

                            vel[cell][0] = conserved_variables[cell][RHO_U_INDEX]/
                                           rho[cell];
                            vel[cell][1] = conserved_variables[cell][RHO_V_INDEX]/
                                           rho[cell];
                         });
}


// Update auxiliary fields after relaxation
//
template<std::size_t dim>
void TwoScale<dim>::update_auxiliary_fields_post_relaxation() {
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           conserved_variables[cell][RHO_ALPHA1_BAR_INDEX] = alpha1_bar[cell]*rho[cell];

                           alpha2_bar[cell] = 1.0 - alpha1_bar[cell];

                           alpha1[cell] = alpha1_bar[cell]*
                                          (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);

                           rho1[cell] = conserved_variables[cell][M1_INDEX]/
                                        alpha1[cell];

                           alpha2[cell] = 1.0
                                        - alpha1[cell]
                                        - conserved_variables[cell][ALPHA1_D_INDEX];

                           rho2[cell] = conserved_variables[cell][M2_INDEX]/alpha2[cell];

                           p_bar[cell] = alpha1_bar[cell]*EOS_phase1.pres_value(rho1[cell])
                                       + alpha2_bar[cell]*EOS_phase2.pres_value(rho2[cell]);

                           const double c_squared = conserved_variables[cell][M1_INDEX]*EOS_phase1.c_value(rho1[cell])*EOS_phase1.c_value(rho1[cell])
                                                  + conserved_variables[cell][M2_INDEX]*EOS_phase2.c_value(rho2[cell])*EOS_phase2.c_value(rho2[cell]);
                           c[cell] = std::sqrt(c_squared/rho[cell])/
                                     (1.0 - conserved_variables[cell][ALPHA1_D_INDEX]);

                           alpha1_d[cell] = conserved_variables[cell][ALPHA1_D_INDEX];
                         });
}


// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void TwoScale<dim>::save(const fs::path& path,
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
void TwoScale<dim>::run() {
  // Default output arguemnts
  fs::path path        = fs::current_path();
  std::string filename = "FV_two_scale";
  const double dt_save = Tf / static_cast<double>(nfiles);

  // Auxiliary variables to save updated fields
  auto conserved_variables_np1 = samurai::make_field<double, EquationData::NVARS>("conserved_np1", mesh);

  // Create the flux variable
  auto flux = samurai::make_two_scale<decltype(conserved_variables)>(vel, p_bar, c);

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, conserved_variables, vel, alpha1_bar, alpha1, p_bar, c, rho1, rho2);

  // Set initial time step
  using mesh_id_t = typename decltype(mesh)::mesh_id_t;
  double dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());
  double dt = cfl*dx/get_max_lambda();

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

    // Perform mesh adaptation
    auto MRadaptation = samurai::make_MRAdapt(alpha1_d);
    MRadaptation(1e-5, 0, conserved_variables, rho, alpha1_bar, alpha2_bar, alpha1, rho1, alpha2, rho2, p_bar, c, vel);

    // Apply the numerical scheme without relaxation
    samurai::update_ghost_mr(conserved_variables, vel, p_bar, c);
    samurai::update_bc(conserved_variables, vel, p_bar, c);
    auto flux_conserved = flux(conserved_variables);
    conserved_variables_np1.resize();
    conserved_variables_np1 = conserved_variables - dt*flux_conserved;

    std::swap(conserved_variables.array(), conserved_variables_np1.array());

    // Update auxiliary useful fields which are not modified by relaxation
    update_auxiliary_fields_pre_relaxation();

    // Apply relaxation if desired, which will modify alpha1_bar and, consequently, for what
    // concerns next time step, rho_alpha1_bar and p_bar
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                             alpha1_bar[cell] = conserved_variables[cell][RHO_ALPHA1_BAR_INDEX]/rho[cell];
                           });
    if(apply_relax) {
      apply_relaxation();
    }

    // Update auxiliary useful fields
    update_auxiliary_fields_post_relaxation();

    // Compute updated time step
    dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());
    dt = std::min(dt, cfl*dx/get_max_lambda());

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix, conserved_variables, vel, alpha1_bar, alpha1, p_bar, c, rho1, rho2);
    }
  }
}
