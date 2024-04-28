#pragma once
#include <samurai/schemes/fv.hpp>

namespace EquationData {
  // Declare spatial dimension
  static constexpr std::size_t dim = 2;

  // Declare parameter related to surface tension coefficient
  static constexpr double sigma = 1e-2;
  static constexpr double kappa = 1.0;
  static constexpr double Hmax  = 40.0;

  // Declare some parameters related to EOS.
  static constexpr double p0_phase1   = 1e5;
  static constexpr double p0_phase2   = 1e5;

  static constexpr double rho0_phase1 = 1e3;
  static constexpr double rho0_phase2 = 1.0;

  static constexpr double c0_phase1   = 1e1;
  static constexpr double c0_phase2   = 1e1;

  // Use auxiliary variables for the indices for the sake of generality
  static constexpr std::size_t M1_INDEX             = 0;
  static constexpr std::size_t M2_INDEX             = 1;
  static constexpr std::size_t M1_D_INDEX           = 2;
  static constexpr std::size_t ALPHA1_D_INDEX       = 3;
  static constexpr std::size_t SIGMA_D_INDEX        = 4;
  static constexpr std::size_t RHO_ALPHA1_BAR_INDEX = 5;
  static constexpr std::size_t RHO_U_INDEX          = 6;

  // Save also the total number of (scalar) variables
  static constexpr std::size_t NVARS = RHO_U_INDEX + dim;
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
    static constexpr std::size_t field_size = Field::size;
    static_assert(field_size == EquationData::NVARS, "The number of elements in the state does not correpsond to the number of equations");
    static_assert(Field::dim == EquationData::dim, "The spatial dimesions do not match");
    static constexpr std::size_t output_field_size = field_size;
    static constexpr std::size_t stencil_size      = 2;

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    Flux(const BarotropicEOS<>& EOS_phase1,
         const BarotropicEOS<>& EOS_phase2,
         const double eps_,
         const double mod_grad_alpha1_bar_min_); // Constructor which accepts in inputs the equations of state of the two phases

    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d,
                                            const auto& grad_alpha1_bar); // Evaluate the 'continuous' flux for the state q along direction curr_d

  protected:
    const BarotropicEOS<>& phase1;
    const BarotropicEOS<>& phase2;

    const double eps;                     // Tolerance of pure phase to set NaNs
    const double mod_grad_alpha1_bar_min; // Tolerance to compute the unit normal
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const BarotropicEOS<>& EOS_phase1,
                    const BarotropicEOS<>& EOS_phase2,
                    const double eps_,
                    const double mod_grad_alpha1_bar_min_): phase1(EOS_phase1), phase2(EOS_phase2), eps(eps_), mod_grad_alpha1_bar_min(mod_grad_alpha1_bar_min_) {}

  // Evaluate the 'continuous flux'
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q,
                                                                             const std::size_t curr_d,
                                                                             const auto& grad_alpha1_bar) {
    // Sanity check in terms of dimensions
    assert(curr_d < EquationData::dim);

    FluxValue<cfg> res = q;

    // Compute the current velocity
    const auto rho   = q(M1_INDEX) + q(M2_INDEX) + q(M1_D_INDEX);
    const auto vel_d = q(RHO_U_INDEX + curr_d)/rho;

    // Multiply the state the velcoity along the direction of interest
    res(M1_INDEX) *= vel_d;
    res(M2_INDEX) *= vel_d;
    res(M1_D_INDEX) *= vel_d;
    res(ALPHA1_D_INDEX) *= vel_d;
    res(SIGMA_D_INDEX) *= vel_d;
    res(RHO_ALPHA1_BAR_INDEX) *= vel_d;
    for(std::size_t d = 0; d < EquationData::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }

    // Compute and add the contribution due to the pressure
    const auto alpha1_bar = q(RHO_ALPHA1_BAR_INDEX)/rho;
    const auto alpha1     = alpha1_bar*(1.0 - q(ALPHA1_D_INDEX));
    const auto rho1       = (alpha1 > eps) ? q(M1_INDEX)/alpha1 : nan("");
    const auto p1         = phase1.pres_value(rho1);

    const auto alpha2     = 1.0 - alpha1 - q(ALPHA1_D_INDEX);
    const auto rho2       = (alpha2 > eps) ? q(M2_INDEX)/alpha2 : nan("");
    const auto p2         = phase2.pres_value(rho2);

    const auto p_bar      = (alpha1_bar > eps && 1.0 - alpha1_bar > eps) ?
                            alpha1_bar*p1 + (1.0 - alpha1_bar)*p2 :
                            ((alpha1_bar < eps) ? p2 : p1);

    res(RHO_U_INDEX + curr_d) += p_bar;

    // Add the contribution due to surface tension
    const auto mod_grad_alpha1_bar = std::sqrt(xt::sum(grad_alpha1_bar*grad_alpha1_bar)());

    if(mod_grad_alpha1_bar > mod_grad_alpha1_bar_min) {
      const auto n = grad_alpha1_bar/mod_grad_alpha1_bar;

      if(curr_d == 0) {
        res(RHO_U_INDEX) += EquationData::sigma*(n(0)*n(0) - 1.0)*mod_grad_alpha1_bar;
        res(RHO_U_INDEX + 1) += EquationData::sigma*n(0)*n(1)*mod_grad_alpha1_bar;
      }

      if(curr_d == 1) {
        res(RHO_U_INDEX) += EquationData::sigma*n(0)*n(1)*mod_grad_alpha1_bar;
        res(RHO_U_INDEX + 1) += EquationData::sigma*(n(1)*n(1) - 1.0)*mod_grad_alpha1_bar;
      }
    }

    return res;
  }


  /**
    * Implementation of a Rusanov flux
    */
  template<class Field>
  class RusanovFlux: public Flux<Field> {
  public:
    RusanovFlux(const BarotropicEOS<>& EOS_phase1,
                const BarotropicEOS<>& EOS_phase2,
                const double eps_,
                const double mod_grad_alpha1_bar_min_); // Constructor which accepts in inputs the equations of state of the two phases

    FluxValue<typename Flux<Field>::cfg> compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                               const FluxValue<typename Flux<Field>::cfg>& qR,
                                                               const std::size_t curr_d,
                                                               const auto& grad_alpha1_barL,
                                                               const auto& grad_alpha1_barR); // Rusanov flux along direction curr_d

    auto make_two_scale_capillarity(const auto& grad_alpha1_bar); // Compute the flux over all cells
  };

  // Constructor derived from the base class
  //
  template<class Field>
  RusanovFlux<Field>::RusanovFlux(const BarotropicEOS<>& EOS_phase1,
                                  const BarotropicEOS<>& EOS_phase2,
                                  const double eps_,
                                  const double grad_alpha1_bar_min_): Flux<Field>(EOS_phase1, EOS_phase2, eps_, grad_alpha1_bar_min_) {}

  // Implementation of a Rusanov flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg> RusanovFlux<Field>::compute_discrete_flux(const FluxValue<typename Flux<Field>::cfg>& qL,
                                                                                 const FluxValue<typename Flux<Field>::cfg>& qR,
                                                                                 const std::size_t curr_d,
                                                                                 const auto& grad_alpha1_barL,
                                                                                 const auto& grad_alpha1_barR) {
    // Compute the quantities needed for the maximum eigenvalue estimate for the left state
    const auto rho_L        = qL(M1_INDEX) + qL(M2_INDEX) + qL(M1_D_INDEX);
    const auto vel_d_L      = qL(RHO_U_INDEX + curr_d)/rho_L;

    const auto alpha1_bar_L = qL(RHO_ALPHA1_BAR_INDEX)/rho_L;
    const auto alpha1_L     = alpha1_bar_L*(1.0 - qL(ALPHA1_D_INDEX));
    const auto rho1_L       = (alpha1_L > this->eps) ? qL(M1_INDEX)/alpha1_L : nan("");
    const auto alpha2_L     = 1.0 - alpha1_L - qL(ALPHA1_D_INDEX);
    const auto rho2_L       = (alpha2_L > this->eps) ? qL(M2_INDEX)/alpha2_L : nan("");
    const auto c_squared_L  = qL(M1_INDEX)*this->phase1.c_value(rho1_L)*this->phase1.c_value(rho1_L)
                            + qL(M2_INDEX)*this->phase2.c_value(rho2_L)*this->phase2.c_value(rho2_L);
    const auto c_L          = std::sqrt(c_squared_L/rho_L)/(1.0 - qL(ALPHA1_D_INDEX));
    const auto r_L          = EquationData::sigma*std::sqrt(xt::sum(grad_alpha1_barL*grad_alpha1_barL)())/(rho_L*c_L*c_L);

    // Compute the quantities needed for the maximum eigenvalue estimate for the right state
    const auto rho_R        = qR(M1_INDEX) + qR(M2_INDEX) + qR(M1_D_INDEX);
    const auto vel_d_R      = qR(RHO_U_INDEX + curr_d)/rho_R;

    const auto alpha1_bar_R = qR(RHO_ALPHA1_BAR_INDEX)/rho_L;
    const auto alpha1_R     = alpha1_bar_R*(1.0 - qR(ALPHA1_D_INDEX));
    const auto rho1_R       = (alpha1_R > this->eps) ? qR(M1_INDEX)/alpha1_R : nan("");
    const auto alpha2_R     = 1.0 - alpha1_R - qR(ALPHA1_D_INDEX);
    const auto rho2_R       = (alpha2_R > this->eps) ? qR(M2_INDEX)/alpha2_R : nan("");
    const auto c_squared_R  = qR(M1_INDEX)*this->phase1.c_value(rho1_R)*this->phase1.c_value(rho1_R)
                            + qR(M2_INDEX)*this->phase2.c_value(rho2_L)*this->phase2.c_value(rho2_R);
    const auto c_R          = std::sqrt(c_squared_R/rho_R)/(1.0 - qR(ALPHA1_D_INDEX));
    const auto r_R          = EquationData::sigma*std::sqrt(xt::sum(grad_alpha1_barR*grad_alpha1_barR)())/(rho_R*c_R*c_R);

    // Compute the estimate of the eigenvalue considering also the surface tension contribution
    const auto lambda = std::max(std::abs(vel_d_L) + c_L*(1.0 + 0.125*r_L),
                                 std::abs(vel_d_R) + c_R*(1.0 + 0.125*r_R));

    return 0.5*(this->evaluate_continuous_flux(qL, curr_d, grad_alpha1_barL) +
                this->evaluate_continuous_flux(qR, curr_d, grad_alpha1_barR)) - // centered contribution
           0.5*lambda*(qR - qL); // upwinding contribution
  }


  // Implement the contribution of the discrete flux for all the cells in the mesh.
  //
  template<class Field>
  auto RusanovFlux<Field>::make_two_scale_capillarity(const auto& grad_alpha1_bar) {
    FluxDefinition<typename Flux<Field>::cfg> Rusanov_f;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      // First, we need a function to compute the "continuous" flux
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function, in this case a Rusanov flux
        Rusanov_f[d].cons_flux_function = [&](auto& cells, const Field& field)
                                          {
                                            const auto& left  = cells[0];
                                            const auto& right = cells[1];

                                            const auto& qL = field[left];
                                            const auto& qR = field[right];

                                            return compute_discrete_flux(qL, qR, d,
                                                                         grad_alpha1_bar[left], grad_alpha1_bar[right]);
                                          };
    });

    return make_flux_based_scheme(Rusanov_f);
  }

} // end namespace samurai
