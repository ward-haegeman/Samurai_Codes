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

#include <samurai/mr/adapt.hpp>

// Specify the use of this namespace where we just store the indices
// and some parameters related to the equations of state
using namespace EquationData;

// This is the class for the simulation of a two-scale model
//
template<std::size_t dim>
class Advection {
public:
  using Config = samurai::MRConfig<dim>;

  Advection() = default; // Default constructor. This will do nothing
                         // and basically will never be used

  Advection(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
            const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
            std::size_t min_level, std::size_t max_level,
            double Tf_, double cfl_, std::size_t nfiles_ = 100);  // Class constrcutor with the arguments related
                                                                   // to the grid, to the physics.

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

  using Field        = samurai::Field<decltype(mesh), double, 1, false>;
  using Field_Vect   = samurai::Field<decltype(mesh), double, dim, false>;

  double Tf;  // Final time of the simulation
  double cfl; // Courant number of the simulation so as to compute the time step

  std::size_t nfiles; // Number of files desired for output

  Field q; // The variable which stores the advected field

  samurai::Advection_Flux<Field> numerical_flux; // Variable to compute the numerical flux

  Field_Vect vel;

  /*--- Now, it's time to declare some member functions that we will employ ---*/
  void init_variables(); // Routine to initialize the variables (both conserved and auxiliary, this is problem dependent)

  double get_max_lambda() const; // Compute the estimate of the maximum eigenvalue
};


// Implement class constructor
//
template<std::size_t dim>
Advection<dim>::Advection(const xt::xtensor_fixed<double, xt::xshape<dim>>& min_corner,
                          const xt::xtensor_fixed<double, xt::xshape<dim>>& max_corner,
                          std::size_t min_level, std::size_t max_level,
                          double Tf_, double cfl_, std::size_t nfiles_):
  box(min_corner, max_corner), mesh(box, min_level, max_level, {true}),
  Tf(Tf_), cfl(cfl_), nfiles(nfiles_),
  numerical_flux() {
    init_variables();
}


// Initialization of conserved and auxiliary variables
//
template<std::size_t dim>
void Advection<dim>::init_variables() {
  // Create conserved and auxiliary fields
  q = samurai::make_field<double, 1>("q", mesh);

  vel = samurai::make_field<double, dim>("vel", mesh);

  const double xr = 0.2;
  const double xd = 0.4;

  // Initialize the fields with a loop over all cells
  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           const auto center = cell.center();
                           const double x    = center[0];

                           vel[cell] = 0.5;

                           if(x < xd && x > xr) {
                             q[cell] = 0.51;
                           }
                           else {
                             q[cell] = 0.49;
                           }
                         });
}


// Compute the estimate of the maximum eigenvalue for CFL condition
//
template<std::size_t dim>
double Advection<dim>::get_max_lambda() const {
  double res = 0.0;

  samurai::for_each_cell(mesh,
                         [&](const auto& cell)
                         {
                           res = std::max(std::abs(vel[cell]), res);
                         });

  return res;
}


// Save desired fields and info
//
template<std::size_t dim>
template<class... Variables>
void Advection<dim>::save(const fs::path& path,
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
void Advection<dim>::run() {
  // Default output arguemnts
  fs::path path        = fs::current_path();
  std::string filename = "Advection";
  const double dt_save = Tf / static_cast<double>(nfiles);

  // Auxiliary variables to save updated fields
  auto q_np1 = samurai::make_field<double, 1>("q_np1", mesh);

  // Create the flux variable
  auto Advection_flux = numerical_flux.make_flux(vel);

  // Save the initial condition
  const std::string suffix_init = (nfiles != 1) ? "_ite_0" : "";
  save(path, filename, suffix_init, q, vel);

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

    // Apply the numerical scheme
    samurai::update_ghost_mr(q, vel);
    samurai::update_bc(q);
    /*const samurai::Stencil<3, EquationData::dim> stencil = {{-1}, {0}, {1}};
    samurai::for_each_stencil(mesh, stencil, [&](const auto& cells)
                                                {
                                                  const auto& left  = cells[0];
                                                  const auto& mid   = cells[1];
                                                  const auto& right = cells[2];

                                                  if(vel[mid] > 0.0) {
                                                    q_np1[mid] = q[mid] - dt/dx*(vel[mid]*(q[mid] - q[left]));
                                                  }
                                                  else {
                                                    q_np1[mid] = q[mid] - dt/dx*(vel[mid]*(q[right] - q[mid]));
                                                  }
                                                });*/
    auto Adv_Flux = Advection_flux(q);
    q_np1         = q - dt*Adv_Flux;

    std::swap(q.array(), q_np1.array());

    // Compute updated time step
    dx = samurai::cell_length(mesh[mesh_id_t::cells].max_level());
    dt = std::min(dt, cfl*dx/get_max_lambda());

    // Save the results
    if(t >= static_cast<double>(nsave + 1) * dt_save || t == Tf) {
      const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", ++nsave) : "";
      save(path, filename, suffix, q, vel);
    }
  }
}
