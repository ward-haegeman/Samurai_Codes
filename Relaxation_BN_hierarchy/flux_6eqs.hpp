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
  static constexpr std::size_t ALPHA2_RHO2_INDEX    = 2;
  static constexpr std::size_t RHO_U_INDEX          = 3;
  static constexpr std::size_t ALPHA1_RHO1_E1_INDEX = RHO_U_INDEX + dim;
  static constexpr std::size_t ALPHA2_RHO2_E2_INDEX = ALPHA1_RHO1_E1_INDEX + 1;

  static constexpr std::size_t NVARS = ALPHA2_RHO2_E2_INDEX + 1;

  // Parameters related to the EOS for the two phases
  static constexpr double gamma_1    = 2.35;
  static constexpr double pi_infty_1 = 1e9;
  static constexpr double q_infty_1  = 0.0;

  static constexpr double gamma_2    = 1.43;
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

    // Save the mixture density and the velocity along the direction of interest
    const auto rho   = q(ALPHA1_RHO1_INDEX) + q(ALPHA2_RHO2_INDEX);
    const auto vel_d = q(RHO_U_INDEX + curr_d)/rho;
    res(RHO_U_INDEX) *= vel_d;
    if(EquationData::dim > 1) {
      for(std::size_t d = 1; d < EquationData::dim; ++d) {
        res(RHO_U_INDEX + d) *= vel_d;
      }
    }

    // Compute density and pressure of phase 1
    const auto alpha1 = q(ALPHA1_INDEX);
    const auto rho1   = q(ALPHA1_RHO1_INDEX)/alpha1; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1           = q(ALPHA1_RHO1_E1_INDEX)/q(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1 -= 0.5*(q(RHO_U_INDEX + d)/rho)*(q(RHO_U_INDEX + d)/rho);
    }
    const auto pres1  = this->phase1.pres_value(rho1, e1);

    // Compute the flux for the equations "associated" to phase 1
    res(ALPHA1_INDEX) = 0.0;
    res(ALPHA1_RHO1_INDEX) *= vel_d;
    res(ALPHA1_RHO1_E1_INDEX) *= vel_d;
    res(ALPHA1_RHO1_E1_INDEX) += alpha1*pres1*vel_d;

    // Compute density and pressure of phase 2
    const auto alpha2 = 1.0 - alpha1;
    const auto rho2   = q(ALPHA2_RHO2_INDEX)/alpha2; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2           = q(ALPHA2_RHO2_E2_INDEX)/q(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2 -= 0.5*(q(RHO_U_INDEX + d)/rho)*(q(RHO_U_INDEX + d)/rho);
    }
    const auto pres2  = this->phase2.pres_value(rho2, e2);

    // Compute the flux for the equations "associated" to phase 2
    res(ALPHA2_RHO2_INDEX) *= vel_d;
    res(ALPHA2_RHO2_E2_INDEX) *= vel_d;
    res(ALPHA2_RHO2_E2_INDEX) += alpha2*pres2*vel_d;

    // Add the mixture pressure contribution to the momentum equation
    res(RHO_U_INDEX + curr_d) += (alpha1*pres1 + alpha2*pres2);

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
    // Save mixture density and velocity current direction left state
    const auto rhoL   = qL(ALPHA1_RHO1_INDEX) + qL(ALPHA2_RHO2_INDEX);
    const auto velL_d = qL(RHO_U_INDEX + curr_d)/rhoL;

    // Left state phase 1
    const auto rho1L  = qL(ALPHA1_RHO1_INDEX)/qL(ALPHA1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1L          = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto pres1L = this->phase1.pres_value(rho1L, e1L);
    const auto c1L    = this->phase1.c_value(rho1L, pres1L);

    // Left state phase 2
    const auto rho2L  = qL(ALPHA2_RHO2_INDEX)/(1.0 - qL(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L          = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto pres2L = this->phase2.pres_value(rho2L, e2L);
    const auto c2L    = this->phase2.c_value(rho2L, pres2L);

    // Compute frozen speed of sound left state
    const auto Y1L = qL(ALPHA1_RHO1_INDEX)/rhoL;
    const auto cL  = std::sqrt(Y1L*c1L*c1L + (1.0 - Y1L)*c2L*c2L);

    // Save mixture density and velocity current direction left state
    const auto rhoR   = qR(ALPHA1_RHO1_INDEX) + qR(ALPHA2_RHO2_INDEX);
    const auto velR_d = qR(RHO_U_INDEX + curr_d)/rhoR;

    // Right state phase 1
    const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/qR(ALPHA1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1R           = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto pres1R  = this->phase1.pres_value(rho1R, e1R);
    const auto c1R     = this->phase1.c_value(rho1R, pres1R);

    // Right state phase 2
    const auto rho2R   = qR(ALPHA2_RHO2_INDEX)/(1.0 - qR(ALPHA1_INDEX)); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R           = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto pres2R  = this->phase2.pres_value(rho2R, e2R);
    const auto c2R     = this->phase2.c_value(rho2R, pres2R);

    // Compute frozen speed of sound left state
    const auto Y1R = qR(ALPHA1_RHO1_INDEX)/rhoR;
    const auto cR  = std::sqrt(Y1R*c1R*c1R + (1.0 - Y1R)*c2R*c2R);

    const auto lambda = std::max(std::abs(velL_d) + cL, std::abs(velR_d) + cR);

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

    // Zero contribution from continuity and momentum equations
    res(ALPHA1_RHO1_INDEX) = 0.0;
    res(ALPHA2_RHO2_INDEX) = 0.0;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      res(RHO_U_INDEX + d) = 0.0;
    }

    // Compute velocity and mass fractions right state
    const auto rhoR = qR(ALPHA1_RHO1_INDEX) + qR(ALPHA2_RHO2_INDEX);
    const auto Y1R  = qR(ALPHA1_RHO1_INDEX)/rhoR;
    const auto Y2R  = qR(ALPHA2_RHO2_INDEX)/rhoR;
    const auto velR = qR(RHO_U_INDEX + curr_d)/rhoR;

    // Compute mixture density left state
    const auto rhoL = qL(ALPHA1_RHO1_INDEX) + qL(ALPHA2_RHO2_INDEX);

    // Pressure phase 1 left state
    const auto alpha1L = qL(ALPHA1_INDEX);
    const auto rho1L   = qL(ALPHA1_RHO1_INDEX)/alpha1L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1L           = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto p1L     = this->phase1.pres_value(rho1L, e1L);

    // Pressure phase 2 left state
    const auto alpha2L = 1.0 - alpha1L;
    const auto rho2L   = qL(ALPHA2_RHO2_INDEX)/alpha2L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L           = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto p2L     = this->phase2.pres_value(rho2L, e2L);

    // Pressure phase 1 right state
    const auto alpha1R = qR(ALPHA1_INDEX);
    const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/alpha1R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1R           = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto p1R     = this->phase1.pres_value(rho1R, e1R);

    // Pressure phase 2 right state
    const auto alpha2R = 1.0 - alpha1R;
    const auto rho2R   = qR(ALPHA2_RHO2_INDEX)/alpha2R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R           = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto p2R     = this->phase2.pres_value(rho2R, e2R);

    // Build the non conservative flux (a lot of approximations to be checked here)
    res(ALPHA1_INDEX) = -0.5*velR*qL(ALPHA1_INDEX);

    res(ALPHA1_RHO1_E1_INDEX) = 0.5*velR*((Y2R*p1R + Y1R*p2R)*alpha1L + alpha1R*Y2R*p1L - alpha2R*Y1R*p2L);

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

    // Zero contribution from continuity and momentum equations
    res(ALPHA1_RHO1_INDEX) = 0.0;
    res(ALPHA2_RHO2_INDEX) = 0.0;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      res(RHO_U_INDEX + d) = 0.0;
    }

    // Compute velocity and mass fractions left state
    const auto rhoL = qL(ALPHA1_RHO1_INDEX) + qL(ALPHA2_RHO2_INDEX);
    const auto Y1L  = qL(ALPHA1_RHO1_INDEX)/rhoL;
    const auto Y2L  = qL(ALPHA2_RHO2_INDEX)/rhoL;
    const auto velL = qL(RHO_U_INDEX + curr_d)/rhoL;

    // Compute mixture density right state
    const auto rhoR = qR(ALPHA1_RHO1_INDEX) + qR(ALPHA2_RHO2_INDEX);

    // Pressure phase 1 left state
    const auto alpha1L = qL(ALPHA1_INDEX);
    const auto rho1L   = qL(ALPHA1_RHO1_INDEX)/alpha1L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1L           = qL(ALPHA1_RHO1_E1_INDEX)/qL(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto p1L     = this->phase1.pres_value(rho1L, e1L);

    // Pressure phase 2 left state
    const auto alpha2L = 1.0 - alpha1L;
    const auto rho2L   = qL(ALPHA2_RHO2_INDEX)/alpha2L; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2L           = qL(ALPHA2_RHO2_E2_INDEX)/qL(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2L -= 0.5*(qL(RHO_U_INDEX + d)/rhoL)*(qL(RHO_U_INDEX + d)/rhoL);
    }
    const auto p2L     = this->phase2.pres_value(rho2L, e2L);

    // Pressure phase 1 right state
    const auto alpha1R = qR(ALPHA1_INDEX);
    const auto rho1R   = qR(ALPHA1_RHO1_INDEX)/alpha1R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e1R           = qR(ALPHA1_RHO1_E1_INDEX)/qR(ALPHA1_RHO1_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e1R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto p1R     = this->phase1.pres_value(rho1R, e1R);

    // Pressure phase 2 right state
    const auto alpha2R = 1.0 - alpha1R;
    const auto rho2R   = qR(ALPHA2_RHO2_INDEX)/alpha2R; /*--- TODO: Add treatment for vanishing volume fraction ---*/
    auto e2R           = qR(ALPHA2_RHO2_E2_INDEX)/qR(ALPHA2_RHO2_INDEX); /*--- TODO: Add treatment for vanishing volume fraction ---*/
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      e2R -= 0.5*(qR(RHO_U_INDEX + d)/rhoR)*(qR(RHO_U_INDEX + d)/rhoR);
    }
    const auto p2R     = this->phase2.pres_value(rho2R, e2R);

    // Build the non conservative flux (a lot of approximations to be checked here)
    res(ALPHA1_INDEX) = 0.5*velL*qR(ALPHA1_INDEX);

    res(ALPHA1_RHO1_E1_INDEX) = -0.5*velL*((Y2L*p1L + Y1L*p2L)*alpha1R + alpha1L*Y2L*p1R - alpha2L*Y1L*p2R);

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
                                              flux[0] = compute_discrete_flux_right_left(qL, qR, d);
                                              flux[1] = compute_discrete_flux_left_right(qL, qR, d);

                                              return flux;
                                            };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }

} // end namespace samurai

#endif
