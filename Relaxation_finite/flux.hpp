#ifndef flux_hpp
#define flux_hpp

#pragma once
#include <samurai/schemes/fv.hpp>

#include "eos.hpp"

namespace EquationData {
  static constexpr std::size_t dim = 1; /*--- Spatial dimension. It would be ideal to be able to get it
                                              direclty from Field, but I need to move the definition of these indices ---*/

  /*--- Declare suitable static variables for the sake of generalities in the indices ---*/
  static constexpr std::size_t ALPHA1_INDEX         = 0;
  static constexpr std::size_t ALPHA1_RHO1_INDEX    = 1;
  static constexpr std::size_t ALPHA1_RHO1_U1_INDEX = 2;
  static constexpr std::size_t ALPHA1_RHO1_E1_INDEX = 2 + dim;
  static constexpr std::size_t ALPHA2_RHO2_INDEX    = ALPHA1_RHO1_E1_INDEX + 1;
  static constexpr std::size_t ALPHA2_RHO2_U2_INDEX = ALPHA2_RHO2_INDEX + 1;
  static constexpr std::size_t ALPHA2_RHO2_E2_INDEX = ALPHA2_RHO2_U2_INDEX + dim;

  static constexpr std::size_t NVARS = ALPHA2_RHO2_E2_INDEX + 1;

  // Parameters related to the EOS for the two phases
  static constexpr double gamma_1    = 3.0;
  static constexpr double pi_infty_1 = 100.0;
  static constexpr double q_infty_1  = 0.0;

  static constexpr double gamma_2    = 1.4;
  static constexpr double pi_infty_2 = 0.0;
  static constexpr double q_infty_2  = 0.0;
}

namespace samurai {
  using namespace EquationData;

  /**
    * Generic class to compute the flux between a left and right state
    */
  template<class Field>
  class Flux {
  public:
    // Definitions and sanity checks
    static constexpr std::size_t field_size        = Field::size;
    static_assert(field_size == EquationData::NVARS, "The number of elements in the state does not correpsond to the number of equations");
    static_assert(Field::dim == EquationData::dim, "The spatial dimesions do not match");
    static constexpr std::size_t output_field_size = field_size;
    static constexpr std::size_t stencil_size      = 2;

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    Flux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2); // Constructor which accepts in inputs the equations of state of the two phases

    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q, const std::size_t curr_d); // Evaluate the 'continuous' flux for the state q along direction curr_d

  protected:
    const EOS<>& phase1; // Pass it by reference because pure virtual (not so nice, maybe moving to pointers)
    const EOS<>& phase2; // Pass it by reference because pure virtual (not so nice, maybe moving to pointers)
  };


  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2): phase1(EOS_phase1), phase2(EOS_phase2) {}


  // Evaluate the 'continuous flux'
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q, const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < EquationData::dim);

    FluxValue<cfg> res = q;

    // Compute density, velocity (along the dimension) and internal energy of phase 1
    const auto alpha1 = q(ALPHA1_INDEX);
    const auto rho1   = q(ALPHA1_RHO1_INDEX)/alpha1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1           = q(ALPHA1_RHO1_E1_INDEX)/q(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1 -= 0.5*(q(ALPHA1_RHO1_U1_INDEX + d)/q(ALPHA1_RHO1_INDEX))*(q(ALPHA1_RHO1_U1_INDEX + d)/q(ALPHA1_RHO1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres1  = this->phase1.pres_value(rho1, e1);
    const auto vel1_d = q(ALPHA1_RHO1_U1_INDEX + curr_d)/q(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/

    // Compute the flux for the equations "associated" to phase 1
    res(ALPHA1_INDEX) = 0.0;
    res(ALPHA1_RHO1_INDEX) *= vel1_d;
    res(ALPHA1_RHO1_U1_INDEX) *= vel1_d;
    if(EquationData::dim > 1) {
      for(std::size_t d = 1; d < EquationData::dim; ++d) {
        res(ALPHA1_RHO1_U1_INDEX + d) *= vel1_d;
      }
    }
    res(ALPHA1_RHO1_U1_INDEX + curr_d) += alpha1*pres1;
    res(ALPHA1_RHO1_E1_INDEX) *= vel1_d;
    res(ALPHA1_RHO1_E1_INDEX) += alpha1*pres1*vel1_d;

    // Compute density, velocity (along the dimension) and internal energy of phase 2
    const auto alpha2 = 1.0 - alpha1;
    const auto rho2   = q(ALPHA2_RHO2_INDEX)/alpha2; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2           = q(ALPHA2_RHO2_E2_INDEX)/q(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2 -= 0.5*(q(ALPHA2_RHO2_U2_INDEX + d)/q(ALPHA2_RHO2_INDEX))*(q(ALPHA2_RHO2_U2_INDEX + d)/q(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres2  = this->phase2.pres_value(rho2, e2);
    const auto vel2_d = q(ALPHA2_RHO2_U2_INDEX + curr_d)/q(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/

    // Compute the flux for the equations "associated" to phase 2
    res(ALPHA2_RHO2_INDEX) *= vel2_d;
    res(ALPHA2_RHO2_U2_INDEX) *= vel2_d;
    if(EquationData::dim > 1) {
      for(std::size_t d = 1; d < EquationData::dim; ++d) {
        res(ALPHA2_RHO2_U2_INDEX + d) *= vel2_d;
      }
    }
    res(ALPHA2_RHO2_U2_INDEX + curr_d) += alpha2*pres2;
    res(ALPHA2_RHO2_E2_INDEX) *= vel2_d;
    res(ALPHA2_RHO2_E2_INDEX) += alpha2*pres2*vel2_d;

    return res;
  }


  /**
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RusanovFlux: public Flux<Field> {
  public:
    RusanovFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2); // Constructor which accepts in inputs the equations of state of the two phases

    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d); // Rusanov flux along direction d

    auto make_flux(); // Compute the flux over all cells
  };


  // Constructor derived from base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2): Flux<Field>(EOS_phase1, EOS_phase2) {}


  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> RusanovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                 std::size_t curr_d) {
    // Left state phase 1
    const auto vel1L_d = qL(ALPHA1_RHO1_U1_INDEX + curr_d)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho1L   = qL(ALPHA1_RHO1_INDEX)/qL(ALPHA1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1L           = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1L -= 0.5*(qL(ALPHA1_RHO1_U1_INDEX + d)/qL(ALPHA1_RHO1_INDEX))*(qL(ALPHA1_RHO1_U1_INDEX + d)/qL(ALPHA1_RHO1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres1L  = this->phase1.pres_value(rho1L, e1L);
    const auto c1L     = this->phase1.c_value(rho1L, pres1L);

    // Left state phase 2
    const auto vel2L_d = qL(ALPHA2_RHO2_U2_INDEX + curr_d)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2L   = qL(ALPHA2_RHO2_INDEX)/(1.0 - qL(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L           = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres2L  = this->phase2.pres_value(rho2L, e2L);
    const auto c2L     = this->phase2.c_value(rho2L, pres2L);

    // Right state phase 1
    const auto vel1R_d = qR(ALPHA1_RHO1_U1_INDEX + curr_d)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/qR(ALPHA1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1R           = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1R -= 0.5*(qR(ALPHA1_RHO1_U1_INDEX + d)/qR(ALPHA1_RHO1_INDEX))*(qR(ALPHA1_RHO1_U1_INDEX + d)/qR(ALPHA1_RHO1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres1R  = this->phase1.pres_value(rho1R, e1R);
    const auto c1R     = this->phase1.c_value(rho1R, pres1R);

    // Right state phase 2
    const auto vel2R_d = qR(ALPHA2_RHO2_U2_INDEX + curr_d)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2R   = qR(ALPHA2_RHO2_INDEX)/(1.0 - qR(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R           = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pres2R  = this->phase2.pres_value(rho2R, e2R);
    const auto c2R     = this->phase2.c_value(rho2R, pres2R);

    const auto lambda = std::max(std::max(std::abs(vel1L_d) + c1L, std::abs(vel1R_d) + c1R),
                                 std::max(std::abs(vel2L_d) + c2L, std::abs(vel2R_d) + c2R));

    return 0.5*(this->evaluate_continuous_flux(qL, curr_d) + this->evaluate_continuous_flux(qR, curr_d)) - // centered contribution
           0.5*lambda*(qR - qL); // upwinding contribution
  }


  // Implement the contribution of the discrete flux for all the cells in the mesh.
  //
  template<class Field>
  auto RusanovFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function
        discrete_flux[d].cons_flux_function = [&](auto& cells, const Field& field)
                                              {
                                                const auto& left  = cells[0];
                                                const auto& right = cells[1];

                                                const auto& qL = field[left];
                                                const auto& qR = field[right];

                                                return compute_discrete_flux(qL, qR, d);
                                              };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }


  /**
    * Implementation of the non-conservative flux
    */
  template<class Field>
  class NonConservativeFlux: public Flux<Field> {
  public:
    NonConservativeFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2); // Constructor which accepts in inputs the equations of state of the two phases

    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux_left_right(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                          const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                          const std::size_t curr_d); // Non-conservative flux from left to right

    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux_right_left(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                          const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                          const std::size_t curr_d); // Non-conservative flux from right to left

    auto make_flux(); // Compute the flux over all cells
  };


  // Constructor derived from base class
  //
  template<class Field>
  NonConservativeFlux<Field>::NonConservativeFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2): Flux<Field>(EOS_phase1, EOS_phase2) {}


  // Implementation of a non-conservative flux from left to right
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> NonConservativeFlux<Field>::compute_discrete_flux_left_right(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                                    const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                                    std::size_t curr_d) {
    FluxValue<typename Flux<Field>::cfg> res;

    // Zero contribution from continuity equations
    res(ALPHA1_RHO1_INDEX) = 0.0;
    res(ALPHA2_RHO2_INDEX) = 0.0;

    // Interfacial velocity and interfacial pressure computed from left state
    const auto velIL = qL(ALPHA1_RHO1_U1_INDEX + curr_d)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2L = qL(ALPHA2_RHO2_INDEX)/(1.0 - qL(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L         = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pIL   = this->phase2.pres_value(rho2L, e2L);

    // Interfacial velocity and interfacial pressure computed from right state
    const auto velIR = qR(ALPHA1_RHO1_U1_INDEX + curr_d)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2R = qR(ALPHA2_RHO2_INDEX)/(1.0 - qR(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R         = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pIR   = this->phase2.pres_value(rho2R, e2R);

    // Build the non conservative flux (Bassi-Rebay formulation)
    res(ALPHA1_INDEX) = (0.5*(velIL*qL(ALPHA1_INDEX) + velIR*qR(ALPHA1_INDEX)) -
                         0.5*(velIL + velIR)*qL(ALPHA1_INDEX));

    res(ALPHA1_RHO1_U1_INDEX + curr_d) = -(0.5*(pIL*qL(ALPHA1_INDEX) + pIR*qR(ALPHA1_INDEX)) -
                                           0.5*(pIL + pIR)*qL(ALPHA1_INDEX));
    res(ALPHA2_RHO2_U2_INDEX + curr_d) = -res(ALPHA1_RHO1_U1_INDEX + curr_d);

    res(ALPHA1_RHO1_E1_INDEX) = -(0.5*(pIL*velIL*qL(ALPHA1_INDEX) + pIR*velIR*qR(ALPHA1_INDEX)) -
                                  0.5*(pIL*velIL + pIR*velIR)*qL(ALPHA1_INDEX));
    res(ALPHA2_RHO2_E2_INDEX) = -res(ALPHA1_RHO1_E1_INDEX);

    return res;
  }


  // Implementation of a non-conservative flux from right to left
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> NonConservativeFlux<Field>::compute_discrete_flux_right_left(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                                    const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                                    std::size_t curr_d) {
    FluxValue<typename Flux<Field>::cfg> res;

    // Zero contribution from continuity equations
    res(ALPHA1_RHO1_INDEX) = 0.0;
    res(ALPHA2_RHO2_INDEX) = 0.0;

    // Interfacial velocity and interfacial pressure computed from left state
    const auto velIL = qL(ALPHA1_RHO1_U1_INDEX + curr_d)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2L = qL(ALPHA2_RHO2_INDEX)/(1.0 - qL(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L         = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pIL   = this->phase2.pres_value(rho2L, e2L);

    // Interfacial velocity and interfacial pressure computed from right state
    const auto velIR = qR(ALPHA1_RHO1_U1_INDEX + curr_d)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto rho2R = qR(ALPHA2_RHO2_INDEX)/(1.0 - qR(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R         = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto pIR   = this->phase2.pres_value(rho2R, e2R);

    // Build the non conservative flux (Bassi-Rebay formulation)
    res(ALPHA1_INDEX) = -(0.5*(velIL*qL(ALPHA1_INDEX) + velIR*qR(ALPHA1_INDEX)) -
                          0.5*(velIL + velIR)*qR(ALPHA1_INDEX));

    res(ALPHA1_RHO1_U1_INDEX + curr_d) = (0.5*(pIL*qL(ALPHA1_INDEX) + pIR*qR(ALPHA1_INDEX)) -
                                          0.5*(pIL + pIR)*qR(ALPHA1_INDEX));
    res(ALPHA2_RHO2_U2_INDEX + curr_d) = -res(ALPHA1_RHO1_U1_INDEX + curr_d);

    res(ALPHA1_RHO1_E1_INDEX) = (0.5*(pIL*velIL*qL(ALPHA1_INDEX) + pIR*velIR*qR(ALPHA1_INDEX)) -
                                 0.5*(pIL*velIL + pIR*velIR)*qR(ALPHA1_INDEX));
    res(ALPHA2_RHO2_E2_INDEX) = -res(ALPHA1_RHO1_E1_INDEX);

    return res;
  }


  // Implement the contribution of the discrete flux for all the cells in the mesh.
  //
  template<class Field>
  auto NonConservativeFlux<Field>::make_flux() {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" non-conservative flux function
        discrete_flux[d].flux_function = [&](auto& cells, const Field& field)
                                            {
                                              const auto& left  = cells[0];
                                              const auto& right = cells[1];

                                              const auto& qL = field[left];
                                              const auto& qR = field[right];

                                              samurai::FluxValuePair<typename Flux<Field>::cfg> flux;
                                              flux[0] = compute_discrete_flux_left_right(qL, qR, d);
                                              flux[1] = compute_discrete_flux_right_left(qL, qR, d);

                                              return flux;
                                            };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }


  /**
    * Implementation of the flux based on Suliciu-type relaxation
    */
  template<class Field>
  class RelaxationFlux: public Flux<Field> {
  public:
    RelaxationFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2); // Constructor which accepts in inputs the equations of state of the two phases

    void compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                               const FluxValue<typename Flux<Field>::cfg>& qR,
                               const std::size_t curr_d,
                               FluxValue<typename Flux<Field>::cfg>& F_minus,
                               FluxValue<typename Flux<Field>::cfg>& F_plus,
                               double& c); // Compute discrete flux

    auto make_flux(double& c); // Compute the flux over all cells.
                               // The input argument is employed to compute the Courant number

  private:
    template<typename T>
    inline T M0(const T nu, const T Me) const;

    template<typename T>
    inline T psi(const T u_star, const T a, const T alphaL, const T alphaR, const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const;

    template<typename T>
    inline T Psi(const T u_star, const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis,
                                 const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis) const;

    template<typename T>
    inline T dM0_dMe(const T nu, const T Me) const;

    template<typename T>
    inline T dpsi_dustar(const T u_star, const T a, const T alphaL, const T alphaR, const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const;

    template<typename T>
    inline T dPsi_dustar(const T u_star, const T a1, const T alpha1L, const T alpha1R,
                                         const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis) const;

    template<typename T>
    T Newton(const T rhs, const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis, const T tau1L_diesis, const T tau1R_diesis,
                          const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis, const double eps) const;

    template<typename T>
    void Riemann_solver_phase_vI(const T xi,
                                 const T alphaL, const T alphaR, const T tauL, const T tauR, const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                 const T a, const T u_star,
                                 T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                 T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p);

    template<typename T>
    void Riemann_solver_phase_pI(const T xi,
                                 const T alphaL, const T alphaR, const T tauL, const T tauR, const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                 const T w_diesis, const T tauL_diesis, const T tauR_diesis, const T a,
                                 T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                 T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p);
  };


  // Constructor derived from base class
  //
  template<class Field>
  RelaxationFlux<Field>::RelaxationFlux(const EOS<>& EOS_phase1, const EOS<>& EOS_phase2): Flux<Field>(EOS_phase1, EOS_phase2) {}


  // Implementation of the flux from left to right (F^{+} in Saleh 2012 notation)
  //
  template<class Field>
  void RelaxationFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                    const FluxValue<typename Flux<Field>::cfg>& qR,
                                                    std::size_t curr_d,
                                                    FluxValue<typename Flux<Field>::cfg>& F_minus,
                                                    FluxValue<typename Flux<Field>::cfg>& F_plus,
                                                    double& c) {
    // Compute the relevant variables from left state for phase 1
    const auto alpha1L = qL(ALPHA1_INDEX);
    const auto rho1L   = qL(ALPHA1_RHO1_INDEX)/alpha1L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel1L_d = qL(ALPHA1_RHO1_U1_INDEX + curr_d)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto E1L     = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1L           = E1L;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1L -= 0.5*(qL(ALPHA1_RHO1_U1_INDEX + d)/qL(ALPHA1_RHO1_INDEX))*(qL(ALPHA1_RHO1_U1_INDEX + d)/qL(ALPHA1_RHO1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p1L     = this->phase1.pres_value(rho1L, e1L);

    // Compute the relevant variables from right state for phase 1
    const auto alpha1R = qR(ALPHA1_INDEX);
    const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/alpha1R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel1R_d = qR(ALPHA1_RHO1_U1_INDEX + curr_d)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto E1R     = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1R           = E1R;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1R -= 0.5*(qR(ALPHA1_RHO1_U1_INDEX + d)/qR(ALPHA1_RHO1_INDEX))*(qR(ALPHA1_RHO1_U1_INDEX + d)/qR(ALPHA1_RHO1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p1R     = this->phase1.pres_value(rho1R, e1R);

    // Compute the relevant variables from left state for phase 2
    const auto alpha2L = 1.0 - alpha1L;
    const auto rho2L   = qL(ALPHA2_RHO2_INDEX)/alpha2L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel2L_d = qL(ALPHA2_RHO2_U2_INDEX + curr_d)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto E2L     = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L           = E2L;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX))*(qL(ALPHA2_RHO2_U2_INDEX + d)/qL(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p2L     = this->phase2.pres_value(rho2L, e2L);

    // Compute the relevant variables from right state for phase 2
    const auto alpha2R = 1.0 - alpha1R;
    const auto rho2R   = qR(ALPHA2_RHO2_INDEX)/alpha2R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto vel2R_d = qR(ALPHA2_RHO2_U2_INDEX + curr_d)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    const auto E2R     = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R           = E2R;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX))*(qR(ALPHA2_RHO2_U2_INDEX + d)/qR(ALPHA2_RHO2_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    }
    const auto p2R     = this->phase2.pres_value(rho2R, e2R);

    // Compute first rhs of relaxation related parameters (Whitham's approach)
    auto a1 = std::max(this->phase1.c_value(rho1L, p1L)*rho1L, this->phase1.c_value(rho1R, p1R)*rho1R);
    auto a2 = std::max(this->phase2.c_value(rho2L, p2L)*rho2L, this->phase2.c_value(rho2R, p2R)*rho2R);

    /*--- Compute the transport step solving a non-linear equation with the Newton method ---*/

    // Compute "diesis" state (formulas (3.21) in Saleh ESAIM 2019, starting point for subsonic wave)
    using field_type = decltype(a2);
    field_type vel1_diesis, p1_diesis, tau1L_diesis = 0.0, tau1R_diesis = 0.0; /*--- NOTE: tau denotes the specific volume, i.e. the inverse of the density ---*/
    field_type vel2_diesis, p2_diesis, tau2L_diesis = 0.0, tau2R_diesis = 0.0;

    const double fact = 1.01; // Safety factor
    // Loop to be sure that tau_diesis variables are positive (theorem 3.5, Coquel et al. JCP 2017)
    while(tau1L_diesis <= 0.0 || tau1R_diesis <= 0.0) {
      a1           *= fact;
      vel1_diesis  = 0.5*(vel1L_d + vel1R_d) - 0.5*(p1R - p1L)/a1;
      p1_diesis    = 0.5*(p1R + p1L) - 0.5*a1*(vel1R_d - vel1L_d);
      tau1L_diesis = 1.0/rho1L + (vel1_diesis - vel1L_d)/a1;
      tau1R_diesis = 1.0/rho1R - (vel1_diesis - vel1R_d)/a1;
    }
    while(tau2L_diesis <= 0.0 || tau2R_diesis <= 0.0) {
      a2           *= fact;
      vel2_diesis  = 0.5*(vel2L_d + vel2R_d) - 0.5*(p2R - p2L)/a2;
      p2_diesis    = 0.5*(p2R + p2L) - 0.5*a2*(vel2R_d - vel2L_d);
      tau2L_diesis = 1.0/rho2L + (vel2_diesis - vel2L_d)/a2;
      tau2R_diesis = 1.0/rho2R - (vel2_diesis - vel2R_d)/a2;
    }

    // Update of a1 and a2 so that a solution for u* surely exists
    field_type rhs = 0.0, sup = 0.0, inf = 0.0;
    const double mu = 0.02;
    while(rhs - inf <= mu*(sup - inf) || sup - rhs <= mu*(sup - inf)) {
      if(vel1_diesis - a1*tau1L_diesis > vel2_diesis - a2*tau2L_diesis &&
         vel1_diesis + a1*tau1R_diesis < vel2_diesis + a2*tau2R_diesis) {
        a1	*= a1;
        vel1_diesis	 = 0.5*(vel1L_d + vel1R_d) - 0.5/a1*(p1R - p1L);
        p1_diesis	   = 0.5*(p1R + p1L) - 0.5*a1*(vel1R_d - vel1L_d);
        tau1L_diesis = 1.0/rho1L + 1.0/a1*(vel1_diesis - vel1L_d);
        tau1R_diesis = 1.0/rho1R - 1.0/a1*(vel1_diesis - vel1R_d);
      }
      else {
        if(vel2_diesis - a2*tau2L_diesis > vel1_diesis - a1*tau1L_diesis &&
           vel2_diesis + a2*tau2R_diesis < vel1_diesis + a1*tau1R_diesis) {
          a2	*= 1.01;
          vel2_diesis	 = 0.5*(vel2L_d + vel2R_d) - 0.5/a2*(p2R - p2L);
          p2_diesis	   = 0.5*(p2R + p2L) - 0.5*a2*(vel2R_d - vel2L_d);
          tau2L_diesis = 1.0/rho2L + 1.0/a2*(vel2_diesis - vel2L_d);
          tau2R_diesis = 1.0/rho2R - 1.0/a2*(vel2_diesis - vel2R_d);
        }
        else {
          a1 *= 1.01;
          vel1_diesis	 = 0.5*(vel1L_d + vel1R_d) - 0.5/a1*(p1R - p1L);
          p1_diesis	   = 0.5*(p1R + p1L) - 0.5*a1*(vel1R_d - vel1L_d);
          tau1L_diesis = 1.0/rho1L + 1.0/a1*(vel1_diesis - vel1L_d);
          tau1R_diesis = 1.0/rho1R - 1.0/a1*(vel1_diesis - vel1R_d);

          a2 *= 1.01;
          vel2_diesis	 = 0.5*(vel2L_d + vel2R_d) - 0.5/a2*(p2R - p2L);
          p2_diesis	   = 0.5*(p2R + p2L) - 0.5*a2*(vel2R_d - vel2L_d);
          tau2L_diesis = 1.0/rho2L + 1.0/a2*(vel2_diesis - vel2L_d);
          tau2R_diesis = 1.0/rho2R - 1.0/a2*(vel2_diesis - vel2R_d);
        }
      }

      // Compute the rhs of the equation for u*
      rhs = -p1_diesis*(alpha1R-alpha1L) -p2_diesis*(alpha2R-alpha2L);

      // Limits on u* so that the relative Mach number is below one
      const auto cLmax = std::max(vel1_diesis - a1*tau1L_diesis, vel2_diesis - a2*tau2L_diesis);
      const auto cRmin = std::min(vel1_diesis + a1*tau1R_diesis, vel2_diesis + a2*tau2R_diesis);

      // Bounds on the function Psi
      inf = Psi(cLmax, a1, alpha1L, alpha1R, vel1_diesis,
                       a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);
      sup = Psi(cRmin, a1, alpha1L, alpha1R, vel1_diesis,
                       a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);

    }

    // Look for u* in the interval [cLmax, cRmin] such that Psi(u*) = rhs
    const double eps   = 1e-7;
  	const auto uI_star = Newton(rhs, a1, alpha1L, alpha1R, vel1_diesis, tau1L_diesis, tau1R_diesis,
                                     a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis, eps);

    // Compute the "fluxes"
    field_type alpha1_m, tau1_m, u1_m, p1_m, E1_m,
               alpha1_p, tau1_p, u1_p, p1_p, E1_p,
               alpha2_m, tau2_m, u2_m, p2_m, E2_m, w2_m,
               alpha2_p, tau2_p, u2_p, p2_p, E2_p, w2_p;
    Riemann_solver_phase_pI(-uI_star,
                            alpha2L, alpha2R, 1.0/rho2L, 1.0/rho2R, vel2L_d - uI_star, vel2R_d - uI_star,
                            p2L, p2R, E2L - (vel2L_d - uI_star)*uI_star - 0.5*uI_star*uI_star, E2R - (vel2R_d - uI_star)*uI_star - 0.5*uI_star*uI_star,
                            vel2_diesis - uI_star, tau2L_diesis, tau2R_diesis, a2,
                            alpha2_m, tau2_m, w2_m, p2_m, E2_m,
                            alpha2_p, tau2_p, w2_p, p2_p, E2_p);
    u2_m = w2_m + uI_star;
  	E2_m += (u2_m - uI_star)*uI_star + 0.5*uI_star*uI_star;
    u2_p = w2_p + uI_star;
  	E2_p += (u2_p - uI_star)*uI_star + 0.5*uI_star*uI_star;
    Riemann_solver_phase_vI(0.0,
                            alpha1L, alpha1R, 1.0/rho1L, 1.0/rho1R, vel1L_d, vel1R_d, p1L, p1R, E1L, E1R,
                            a1, uI_star,
                            alpha1_m, tau1_m, u1_m, p1_m, E1_m,
                            alpha1_p, tau1_p, u1_p, p1_p, E1_p);

    // Build the "fluxes"
    F_minus(ALPHA1_INDEX) = 0.0;

    F_minus(ALPHA1_RHO1_INDEX)             = alpha1_m/tau1_m*u1_m;
    F_minus(ALPHA1_RHO1_U1_INDEX + curr_d) = alpha1_m/tau1_m*u1_m*u1_m + alpha1_m*p1_m;
    F_minus(ALPHA1_RHO1_E1_INDEX)          = alpha1_m/tau1_m*E1_m*u1_m + alpha1_m*p1_m*u1_m;

    F_minus(ALPHA2_RHO2_INDEX)             = alpha2_m/tau2_m*u2_m;
    F_minus(ALPHA2_RHO2_U2_INDEX + curr_d) = alpha2_m/tau2_m*u2_m*u2_m + alpha2_m*p2_m;
    F_minus(ALPHA2_RHO2_E2_INDEX)          = alpha2_m/tau2_m*E2_m*u2_m + alpha2_m*p2_m*u2_m;

    F_plus(ALPHA1_INDEX) = 0.0;

    F_plus(ALPHA1_RHO1_INDEX)             = alpha1_p/tau1_p*u1_p;
    F_plus(ALPHA1_RHO1_U1_INDEX + curr_d) = alpha1_p/tau1_p*u1_p*u1_p + alpha1_p*p1_p;
    F_plus(ALPHA1_RHO1_E1_INDEX)          = alpha1_p/tau1_p*E1_p*u1_p + alpha1_p*p1_p*u1_p;

    F_plus(ALPHA2_RHO2_INDEX)             = alpha2_p/tau2_p*u2_p;
    F_plus(ALPHA2_RHO2_U2_INDEX + curr_d) = alpha2_p/tau2_p*u2_p*u2_p + alpha2_p*p2_p;
    F_plus(ALPHA2_RHO2_E2_INDEX)          = alpha2_p/tau2_p*E2_p*u2_p + alpha2_p*p2_p*u2_p;

    // Focus on non-conservative term
    const auto pidxalpha2	= p2_diesis*(alpha2R - alpha2L) + psi(uI_star, a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);

    if(uI_star < 0.0) {
      F_minus(ALPHA1_INDEX) -= -uI_star*(alpha1R - alpha1L);

      F_minus(ALPHA1_RHO1_U1_INDEX) -= -pidxalpha2;
      F_minus(ALPHA1_RHO1_E1_INDEX) -= -uI_star*pidxalpha2;

      F_minus(ALPHA2_RHO2_U2_INDEX) -= pidxalpha2;
      F_minus(ALPHA2_RHO2_E2_INDEX) -= uI_star*pidxalpha2;
    }
    else {
      F_plus(ALPHA1_INDEX) += -uI_star*(alpha1R - alpha1L);

      F_plus(ALPHA1_RHO1_U1_INDEX) += -pidxalpha2;
      F_plus(ALPHA1_RHO1_E1_INDEX) += -uI_star*pidxalpha2;

      F_plus(ALPHA2_RHO2_U2_INDEX) += pidxalpha2;
      F_plus(ALPHA2_RHO2_E2_INDEX) += uI_star*pidxalpha2;
    }

    c = std::max(c, std::max(std::max(std::abs(vel1L_d - a1/rho1L), std::abs(vel1R_d + a1/rho1R)),
                             std::max(std::abs(vel2L_d - a2/rho2L), std::abs(vel2R_d + a2/rho2R))));
  }


  // Implement the contribution of the discrete flux for all the cells in the mesh.
  //
  template<class Field>
  auto RelaxationFlux<Field>::make_flux(double& c) {
    FluxDefinition<typename Flux<Field>::cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" non-conservative flux function
        discrete_flux[d].flux_function = [&](auto& cells, const Field& field)
                                            {
                                              const auto& left  = cells[0];
                                              const auto& right = cells[1];

                                              const auto& qL = field[left];
                                              const auto& qR = field[right];

                                              FluxValue<typename Flux<Field>::cfg> F_minus,
                                                                                   F_plus;

                                              compute_discrete_flux(qL, qR, d, F_minus, F_plus, c);

                                              samurai::FluxValuePair<typename Flux<Field>::cfg> flux;
                                              flux[0] = F_minus;
                                              flux[1] = -F_plus;

                                              return flux;
                                            };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }


  // Implement M0 function (3.312 Saleh 2012, 3.30 Saleh ESAIM 2019)
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::M0(const T nu, const T Me) const {
    return 4.0/(nu + 1.0)*Me/((1.0 + Me*Me)*(1.0 + std::sqrt(std::abs(1.0 - 4.0*nu/((nu + 1.0)*(nu + 1.0))*4.0*Me*Me/((1.0 + Me*Me)*(1.0 + Me*Me))))));
  }


  // Implement psi function (Saleh 2012 ??)
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::psi(const T u_star, const T a, const T alphaL, const T alphaR, const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const {
    if(u_star <= vel_diesis) {
      return a*(alphaL + alphaR)*(u_star - vel_diesis) + 2.0*a*a*alphaL*tauL_diesis*M0(alphaL/alphaR, (vel_diesis - u_star)/(a*tauL_diesis));
    }

    return -psi(-u_star, a, alphaR, alphaL, -vel_diesis, tauR_diesis, tauL_diesis);
  }


  // Implement Psi function (3.3.15 Saleh 2012 ??)
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::Psi(const T u_star, const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis,
                                                      const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis) const {
    return a1*(alpha1L + alpha1R)*(u_star - vel1_diesis) + psi(u_star, a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);
  }


  // Implement the derivative of M0 w.r.t Me for the Newton method
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::dM0_dMe(const T nu, const T Me) const {
    const T w = (1.0 - Me)/(1.0 + Me);

    return 4.0/(nu + 1.0)*w/((1.0 + w*w)*(1.0 + w*w))*(1.0 + w)*(1.0 + w)/
           (1.0 - 4.0*nu/((nu + 1.0)*(nu + 1.0))*(1.0 - w*w)*(1.0 - w*w)/((1.0 + w*w)*(1.0 + w*w)) +
            std::sqrt(std::abs(1.0 - 4.0*nu/((nu + 1.0)*(nu + 1.0))*(1.0 - w*w)*(1.0 - w*w)/((1.0 + w*w)*(1.0 + w*w)))));
  }


  // Implement the derivative of psi w.r.t. u* for the Newton method
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::dpsi_dustar(const T u_star, const T a, const T alphaL, const T alphaR, const T vel_diesis, const T tauL_diesis, const T tauR_diesis) const {
    if(u_star <= vel_diesis) {
      return a*(alphaL + alphaR) - 2.0*a*alphaL*dM0_dMe(alphaL/alphaR, (vel_diesis - u_star)/(a*tauL_diesis));
    }

    return a*(alphaL + alphaR) - 2.0*a*alphaR*dM0_dMe(alphaR/alphaL, (vel_diesis - u_star)/(a*tauR_diesis));
  }


  // Implement the derivative of Psi w.r.t. u* for the Newton method
  //
  template<class Field>
  template<typename T>
  inline T RelaxationFlux<Field>::dPsi_dustar(const T u_star, const T a1, const T alpha1L, const T alpha1R,
                                                              const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis) const {
    return a1*(alpha1L + alpha1R) + dpsi_dustar(u_star, a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis);
  }


  // Newton method to compute u*
  //
  template<class Field>
  template<typename T>
  T RelaxationFlux<Field>::Newton(const T rhs, const T a1, const T alpha1L, const T alpha1R, const T vel1_diesis, const T tau1L_diesis, const T tau1R_diesis,
                                               const T a2, const T alpha2L, const T alpha2R, const T vel2_diesis, const T tau2L_diesis, const T tau2R_diesis, const double eps) const {
    if(alpha1L == alpha1R) {
      return vel1_diesis;
    }
    else {
      unsigned int iter = 0;
      const T xl = std::max(vel1_diesis - a1*tau1L_diesis, vel2_diesis - a2*tau2L_diesis);
      const T xr = std::min(vel1_diesis + a1*tau1R_diesis, vel2_diesis + a2*tau2R_diesis);

      T u_star = 0.5*(xl + xr);

      while(iter < 1000 &&
            std::abs(Psi(u_star, a1, alpha1L, alpha1R, vel1_diesis,
                                 a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis) - rhs) > eps) {
        ++iter;

        u_star -= (Psi(u_star, a1, alpha1L, alpha1R, vel1_diesis,
                               a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis) - rhs)/
                  (dPsi_dustar(u_star, a1, alpha1L, alpha1R,
                                       a2, alpha2L, alpha2R, vel2_diesis, tau2L_diesis, tau2R_diesis));
      }

      // Safety check
      if(iter == 1000) {
        std::cout << "Newton method not converged." << std::endl;
        exit(0);
      }

      return u_star;
    }
  }


  // Riemann solver for the phase associated to the interfacial velocity
  //
  template<class Field>
  template<typename T>
  void RelaxationFlux<Field>::Riemann_solver_phase_vI(const T xi,
                                                      const T alphaL, const T alphaR, const T tauL, const T tauR, const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                                      const T a, const T u_star,
                                                      T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                                      T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p) {
    if(xi < wL-a*tauL) {
  		alpha_m = alphaL;
  		tau_m	  = tauL;
  		w_m	    = wL;
  		pres_m	= pL;
      E_m     = EL;

  		alpha_p = alphaL;
  		tau_p	  = tauL;
  		w_p	    = wL;
  		pres_p	= pL;
      E_p     = EL;
  	}
  	else {
  		if(xi == wL-a*tauL)	{
  			alpha_m = alphaL;
  			tau_m	  = tauL;
  			w_m	    = wL;
  			pres_m	= pL;
        E_m     = EL;

  			alpha_p = alphaL;
  			tau_p	  = tauL + 1./a*(u_star - wL);
  			w_p	    = u_star;
  			pres_p	= pL + a*(wL - u_star);
        E_p	    = EL - 1.0/a*(pres_p*w_p - pL*wL);
  		}
  		else {
  			if(xi > wL - a*tauL && xi < u_star)	{
  				alpha_m = alphaL;
  				tau_m	  = tauL + 1.0/a*(u_star - wL);
  				w_m	    = u_star;
  				pres_m	= pL + a*(wL - u_star);
          E_m	    = EL -1.0/a*(pres_m*w_m - pL*wL);

  				alpha_p = alphaL;
  				tau_p	  = tauL + 1.0/a*(u_star - wL);
  				w_p	    = u_star;
  				pres_p	= pL + a*(wL - u_star);
          E_p	    = EL - 1.0/a*(pres_p*w_p - pL*wL);
  			}
  			else {
  				if(xi == u_star) {
  					alpha_m = alphaL;
  					tau_m	  = tauL + 1.0/a*(u_star - wL);
  					w_m	    = u_star;
  					pres_m	= pL + a*(wL - u_star);
            E_m	    = EL - 1.0/a*(pres_m*w_m - pL*wL);

  					alpha_p = alphaR;
  					tau_p	  = tauR - 1.0/a*(u_star - wR);
  					w_p	    = u_star;
  					pres_p	= pR - a*(wR - u_star);
            E_p	    = ER + 1.0/a*(pres_p*w_p - pR*wR);
  				}
  				else {
  					if(xi > u_star && xi < wR + a*tauR)	{
  						alpha_m = alphaR;
  						tau_m	  = tauR - 1.0/a*(u_star - wR);
  						w_m	    = u_star;
  						pres_m	= pR - a*(wR - u_star);
              E_m	    = ER + 1.0/a*(pres_m*w_m - pR*wR);

  						alpha_p = alphaR;
  						tau_p	  = tauR - 1.0/a*(u_star - wR);
  						w_p	    = u_star;
  						pres_p	= pR - a*(wR - u_star);
              E_p	    = ER + 1.0/a*(pres_p*w_p - pR*wR);
  					}
  					else {
  						if(xi == wR + a*tauR)	{
  							alpha_m = alphaR;
  							tau_m	  = tauR - 1.0/a*(u_star - wR);
  							w_m	    = u_star;
  							pres_m	= pR - a*(wR - u_star);
                E_m	    = ER + 1.0/a*(pres_m*w_m - pR*wR);

  							alpha_p = alphaR;
  							tau_p	  = tauR;
  							w_p	    = wR;
  							pres_p	= pR;
                E_p	    = ER;
  						}
  						else {
  							alpha_m = alphaR;
  							tau_m	  = tauR;
  							w_m	    = wR;
  							pres_m	= pR;
                E_m	    = ER;

  							alpha_p = alphaR;
  							tau_p	  = tauR;
  							w_p	    = wR;
  							pres_p	= pR;
                E_p	    = ER;
  						}
  					}
  				}
  			}
  		}
  	}
  }


  // Riemann solver for the phase associated to the interfacial pressure
  //
  template<class Field>
  template<typename T>
  void RelaxationFlux<Field>::Riemann_solver_phase_pI(const T xi,
                                                      const T alphaL, const T alphaR, const T tauL, const T tauR, const T wL, const T wR, const T pL, const T pR, const T EL, const T ER,
                                                      const T w_diesis, const T tauL_diesis, const T tauR_diesis, const T a,
                                                      T& alpha_m, T& tau_m, T& w_m, T& pres_m, T& E_m,
                                                      T& alpha_p, T& tau_p, T& w_p, T& pres_p, T& E_p) {
    const T nu  = alphaL/alphaR;
    const T ML  = wL/(a*tauL);
    const T MdL = w_diesis/(a*tauL_diesis);

    T M;
    T Mzero;
    const T mu = 0.9;
    const T t  = tauR_diesis/tauL_diesis;

  	if(w_diesis > 0.0)	{
  		if(ML <  1.0) {
         /*--- Configuration <1,2> subsonic.
  	           Computation of M which parametrisez the whole solution ---*/
  			Mzero = 4.0/(nu + 1.0)*MdL/((1.0 + MdL*MdL)*(1.0 + std::sqrt(std::abs(1.0 - 4.0*nu/((nu + 1.0)*(nu + 1.0))*4.0*MdL*MdL/((1.0 + MdL*MdL)*(1.0 + MdL*MdL))))));

  			if(mu*tauR_diesis <= tauR_diesis + tauL_diesis*(MdL + nu*Mzero)/(1.+nu*Mzero)){
          M = Mzero;
        }
        else {
          /*--- Add the required amount of energy dissipation ---*/
          M = 1.0/nu*(MdL + t*(1.0 - mu))/(1.0 - t*(1.0 - mu));
  			}
  		}

  		if(xi < wL - a*tauL) {
  			alpha_m = alphaL;
  			tau_m	  = tauL;
  			w_m	    = wL;
  			pres_m	= pL;
        E_m	    = EL;

  			alpha_p = alphaL;
  			tau_p	  = tauL;
  			w_p	    = wL;
  			pres_p	= pL;
        E_p	    = EL;
  		}
  		else {
  			if(xi == wL - a*tauL)	{
  				alpha_m = alphaL;
  				tau_m	  = tauL;
  				w_m	    = wL;
  				pres_m	= pL;
          E_m	    = EL;

  				alpha_p = alphaL;
  				tau_p	  = tauL_diesis*(1.0 - MdL)/(1.0 - M);
  				w_p	    = a*M*tau_p;
  				pres_p	= pL + a*(wL - w_p);
          E_p	    = EL - 1.0/a*(pres_p*w_p - pL*wL);
  			}
  			else {
  				if(xi > wL - a*tauL && xi < 0.0) {
  					alpha_m = alphaL;
  					tau_m	  = tauL_diesis*(1.0 - MdL)/(1.0 - M);
  					w_m	    = a*M*tau_m;
  					pres_m	= pL + a*(wL - w_m);
            E_m	    = EL - 1.0/a*(pres_m*w_m - pL*wL);

  				  alpha_p = alphaL;
  					tau_p	  = tauL_diesis*(1.0 - MdL)/(1.0 - M);
  					w_p	    = a*M*tau_p;
  					pres_p	= pL + a*(wL - w_p);
            E_p	    = EL - 1.0/a*(pres_p*w_p - pL*wL);
  				}
  				else {
  					if(xi == 0.0)	{
  						alpha_m = alphaL;
  						tau_m	  = tauL_diesis*(1.0 - MdL)/(1.0 - M);
  						w_m	    = a*M*tau_m;
  						pres_m	= pL + a*(wL - w_m);
              E_m	    = EL - 1.0/a*(pres_m*w_m - pL*wL);

  						alpha_p = alphaR;
  						tau_p	  = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
  						w_p	    = nu*a*M*tau_p;
  						pres_p	= pL + a*a*(tauL - tau_p);
              E_p	    = E_m - (pres_p*tau_p - pres_m*tau_m);
  					}
  					else {
  						if(xi > 0.0 && xi < nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M)) {
  							/*--- Computations of E_m and E_p ---*/
  						  alpha_m = alphaL;
                tau_m	  = tauL_diesis*(1.0 - MdL)/(1.0 - M);
                w_m	    = a*M*tau_m;
                pres_m	= pL + a*(wL - w_m);
                E_m	    = EL - 1.0/a*(pres_m*w_m - pL*wL);

                alpha_p = alphaR;
                tau_p	  = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
                w_p 	  = nu*a*M*tau_p;
                pres_p	= pL + a*a*(tauL - tau_p);
                E_p	    = E_m - (pres_p*tau_p - pres_m*tau_m);

                /*--- Compute the real states ---*/
                alpha_m = alphaR;
  							tau_m	  = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
  							w_m 	  = nu*a*M*tau_m;
  							pres_m	= pL + a*a*(tauL - tau_m);
                E_m     = E_p;

  						  alpha_p = alphaR;
  							tau_p	  = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
  							w_p	    = nu*a*M*tau_p;
  							pres_p	= pL + a*a*(tauL - tau_p);
  						}
  						else {
  							if(xi == nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M))	{
                  /*--- Computations of E_m and E_p ---*/
                  alpha_m = alphaL;
                  tau_m	  = tauL_diesis*(1.0 - MdL)/(1.0 - M);
                  w_m	    = a*M*tau_m;
                  pres_m	= pL + a*(wL - w_m);
                  E_m	    = EL - 1.0/a*(pres_m*w_m - pL*wL);

                  alpha_p = alphaR;
                  tau_p	  = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
                  w_p	    = nu*a*M*tau_p;
                  pres_p	= pL + a*a*(tauL - tau_p);
                  E_p	    = E_m - (pres_p*tau_p - pres_m*tau_m);

                  /*--- Compute the real states ---*/
                  alpha_m = alphaR;
  								tau_m	  = tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
  								w_m	    = nu*a*M*tau_m;
  								pres_m	= pL + a*a*(tauL - tau_m);
                  E_m     = E_p;

  								alpha_p = alphaR;
  								tau_p	  = tauR_diesis + tauL_diesis*(MdL - nu*M)/(1.0 + nu*M);
  								w_p	    = nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
  								pres_p	= pR - a*(wR - w_p);
                  E_p     = ER - 1.0/a*(pR*wR - pres_p*w_p);
  							}
  							else {
  								if(xi > nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M) && xi < wR + a*tauR) {
  									alpha_m = alphaR;
  									tau_m	  = tauR_diesis + tauL_diesis*(MdL - nu*M)/(1.0 + nu*M);
  									w_m	    = nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
  									pres_m	= pR - a*(wR - w_m);
                    E_m     = ER - 1.0/a*(pR*wR - pres_m*w_m);

  									alpha_p = alphaR;
  									tau_p	  = tauR_diesis + tauL_diesis*(MdL - nu*M)/(1.0 + nu*M);
  									w_p	    = nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
  									pres_p	= pR - a*(wR - w_p);
                    E_p     = ER + 1.0/a*(pres_p*w_p - pR*wR);
  								}
  								else {
  									if(xi == wR + a*tauR) {
  										alpha_m = alphaR;
  										tau_m	  = tauR_diesis + tauL_diesis*(MdL - nu*M)/(1.0 + nu*M);
  										w_m	    = nu*a*M*tauL_diesis*(1.0 + MdL)/(1.0 + nu*M);
  										pres_m	= pR - a*(wR - w_m);
                      E_m     = ER + 1.0/a*(pres_m*w_m - pR*wR);

  										alpha_p = alphaR;
  										tau_p	  = tauR;
  										w_p	    = wR;
  										pres_p	= pR;
                      E_p     = ER;
  									}
  									else
  									{
  										alpha_m = alphaR;
  										tau_m	  = tauR;
  										w_m	    = wR;
  										pres_m	= pR;
                      E_m     = ER;

  										alpha_p = alphaR;
  										tau_p	  = tauR;
  										w_p	    = wR;
  										pres_p	= pR;
                      E_p     = ER;
  									}
  								}
  							}
  						}
  					}

  				}
  			}
  		}
  	}
  	else {
  		if(w_diesis < 0.0) {
        Riemann_solver_phase_pI	(-xi,
                                 alphaR, alphaL, tauR, tauL, -wR, -wL, pR, pL, ER, EL,
                                 -w_diesis, tauR_diesis, tauL_diesis, a,
                                 alpha_p, tau_p, w_p, pres_p, E_p, alpha_m, tau_m, w_m, pres_m, E_m);
        w_m = -w_m;
        w_p = -w_p;
      }
      else {
        if(xi < wL - a*tauL) {
  				alpha_m = alphaL;
  				tau_m	  = tauL;
  				w_m	    = wL;
  				pres_m	= pL;
  				E_m	    = EL;

  				alpha_p = alphaL;
  				tau_p	  = tauL;
  				w_p	    = wL;
  				pres_p	= pL;
  				E_p	    = EL;
  			}
  			else {
  				if(xi == wL - a*tauL)	{
  					alpha_m = alphaL;
  					tau_m	  = tauL;
  					w_m	    = wL;
  					pres_m	= pL;
  					E_m	    = EL;

  					alpha_p = alphaL;
  					tau_p	  = tauL_diesis;
  					w_p	    = 0.0;
  					pres_p	= pL + a*(wL - w_p);
  					E_p	    = EL - 1.0/a*(pres_p*w_p - pL*wL);
  				}
  				else {
  					if(xi > wL - a*tauL && xi < 0.0) {
  						alpha_m = alphaL;
  						tau_m	  = tauL_diesis;
  						w_m	    = 0.0;
  						pres_m	= pL + a*(wL - w_m);
  						E_m	    = EL - 1.0/a*(pres_m*w_m - pL*wL);

  						alpha_p = alphaL;
  						tau_p	  = tauL_diesis;
  						w_p	    = 0.0;
  						pres_p	= pL + a*(wL - w_p);
  						E_p	    = EL - 1.0/a*(pres_p*w_p - pL*wL);
  					}
  					else {
  						if(xi == 0.0)	{
  							alpha_m = alphaL;
  							tau_m	  = tauL_diesis;
  							w_m	    = 0.0;
  							pres_m	= pL + a*(wL - w_m);
  							E_m	    = EL - 1.0/a*(pres_m*w_m - pL*wL);

  							alpha_p = alphaR;
  							tau_p	  = tauR_diesis;
  							w_p	    = 0.0;
  							pres_p	= pR - a*(wR - w_p);
  							E_p	    = ER + 1.0/a*(pres_p*w_p - pR*wR);
  						}
  						else {
  							if(xi > 0.0 && xi < wR + a*tauR) {
  								alpha_m = alphaR;
  								tau_m	  = tauR_diesis;
  								w_m	    = 0.0;
  								pres_m	= pR - a*(wR - w_m);
  								E_m	    = ER + 1.0/a*(pres_m*w_m - pR*wR);

  								alpha_p = alphaR;
  								tau_p	  = tauR_diesis;
  								w_p	    = 0.0;
  								pres_p	= pR - a*(wR-w_p);
  								E_p	    = ER + 1.0/a*(pres_p*w_p - pR*wR);
  							}
  							else {
  								if(xi == wR + a*tauR)	{
  									alpha_m = alphaR;
  									tau_m	  = tauR_diesis;
  									w_m	    = 0.0;
  									pres_m	= pR - a*(wR - w_m);
  									E_m	    = ER + 1.0/a*(pres_m*w_m - pR*wR);

  									alpha_p = alphaR;
  									tau_p	  = tauR;
  									w_p	    = wR;
  									pres_p	= pR;
  									E_p	    = ER;
  								}
  								else {
  									alpha_m = alphaR;
  									tau_m	  = tauR;
  									w_m	    = wR;
  									pres_m  = pR;
  									E_m    	= ER;

  									alpha_p = alphaR;
  									tau_p	  = tauR;
  									w_p	    = wR;
  									pres_p	= pR;
  									E_p	    = ER;
  								}
  							}
  						}
  					}
  				}
  			}
  		}
  	}
  }

} // end namespace samurai

#endif
