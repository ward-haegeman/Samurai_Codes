#pragma once
#include <samurai/schemes/fv.hpp>

namespace EquationData {
  // Declare spatial dimension
  static constexpr std::size_t dim = 2;

  // Declare some parameters related to EOS.
  static constexpr double p0_phase1   = 1e5;
  static constexpr double p0_phase2   = 1e5;

  static constexpr double rho0_phase1 = 1.0;
  static constexpr double rho0_phase2 = 1e3;

  static constexpr double c0_phase1   = 3.0;
  static constexpr double c0_phase2   = 15.0;

  // Use auxiliary variables for the indices for the sake of generality
  static constexpr std::size_t M1_INDEX             = 0;
  static constexpr std::size_t M2_INDEX             = 1;
  static constexpr std::size_t M1_D_INDEX           = 2;
  static constexpr std::size_t ALPHA1_D_INDEX       = 3;
  static constexpr std::size_t RHO_ALPHA1_BAR_INDEX = 4;
  static constexpr std::size_t RHO_U_INDEX          = 5;
  static constexpr std::size_t RHO_V_INDEX          = 6;

  // Save also the total number of (scalar) variables
  static constexpr std::size_t NVARS = 5 + dim;
}


namespace samurai {
  using namespace EquationData;

  /**
   * Implementation of discretization of a conservation law with upwind/Rusanov flux
   * along the horizontal direction
   */
  template<class Field, class Field_Vect, class Field_Scalar>
  auto make_two_scale(const Field_Vect& vel, const Field_Scalar& pres, const Field_Scalar& c) {
    static_assert(Field::dim == EquationData::dim, "No mathcing spatial dimension between Field and Data");
    static_assert(Field::dim == Field_Vect::size, "No mathcing spactial dimension in make_two_scale");
    static_assert(Field::size == EquationData::NVARS, "The number of elements in the state does not correpsond to the number of equations");

    static constexpr std::size_t field_size        = Field::size;
    static constexpr std::size_t output_field_size = field_size;
    static constexpr std::size_t stencil_size      = 2;

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    FluxDefinition<cfg> Rusanov_f;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      // First, we need a function to compute the "continuous" flux
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        auto f = [&](const auto& q, const auto& velocity, const auto& pressure)
        {
          FluxValue<cfg> res = q;

          res(M1_INDEX) *= velocity(d);
          res(M2_INDEX) *= velocity(d);
          res(M1_D_INDEX) *= velocity(d);
          res(ALPHA1_D_INDEX) *= velocity(d);
          res(RHO_ALPHA1_BAR_INDEX) *= velocity(d);
          res(RHO_U_INDEX) *= velocity(d);
          res(RHO_V_INDEX) *= velocity(d);

          if constexpr(d == 0) {
            res(RHO_U_INDEX) += pressure;
          }
          if constexpr(d == 1) {
            res(RHO_V_INDEX) += pressure;
          }

          return res;
        };

        // Compute now the "discrete" flux function, in this case a Rusanov flux
        Rusanov_f[d].cons_flux_function = [&](auto& cells, const Field& field)
                                          {
                                            const auto& left  = cells[0];
                                            const auto& right = cells[1];

                                            const auto lambda = std::max(std::max(std::abs(vel[left](d) + c[left]),
                                                                                  std::abs(vel[left](d) - c[left])),
                                                                         std::max(std::abs(vel[right](d) + c[right]),
                                                                                  std::abs(vel[right](d) + c[right])));

                                            return 0.5*(f(field[left], vel[left], pres[left]) + f(field[right], vel[right], pres[right])) -
                                                   0.5*lambda*(field[right] - field[left]);
                                          };
      }
    );

    return make_flux_based_scheme(Rusanov_f);
  }

} // end namespace samurai
