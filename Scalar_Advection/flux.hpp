#ifndef flux_hpp
#define flux_hpp

#pragma once
#include <samurai/schemes/fv.hpp>

namespace EquationData {
  static constexpr std::size_t dim = 1; /*--- Spatial dimension. It would be ideal to be able to get it
                                              direclty from Field, but I need to move the definition of these indices ---*/
}

namespace samurai {
  using namespace EquationData;

  /**
    * Class to compute the flux between a left and right state
    */
  template<class Field>
  class Advection_Flux {
  public:
    // Definitions and sanity checks
    static constexpr std::size_t field_size        = Field::size;
    static_assert(Field::dim == EquationData::dim, "The spatial dimesions do not match");
    static constexpr std::size_t output_field_size = field_size;
    static constexpr std::size_t stencil_size      = 2;

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    Advection_Flux() = default; // Default constructor

    template<class Field_Vel>
    FluxValue<cfg> evaluate_continuos_flux(const FluxValue<cfg>& q,
                                           const Field_Vel& vel,
                                           const std::size_t curr_d); // Evaluate the 'continuous' flux for the state q along direction curr_d

    template<class Field_Vel>
    FluxValue<cfg> compute_discrete_flux_left_right(const FluxValue<cfg>& qL,
                                                    const FluxValue<cfg>& qR,
                                                    const Field_Vel& vel_L,
                                                    const Field_Vel& vel_R,
                                                    const std::size_t curr_d); // advection flux along direction d (left from right contribution)

    template<class Field_Vel>
    FluxValue<cfg> compute_discrete_flux_right_left(const FluxValue<cfg>& qL,
                                                    const FluxValue<cfg>& qR,
                                                    const Field_Vel& vel_L,
                                                    const Field_Vel& vel_R,
                                                    const std::size_t curr_d); // advection flux along direction d (right from left contribution)

    template<class Field_Vect>
    auto make_flux(const Field_Vect& vel); // Compute the flux over all cells
  };


  // Evaluate the 'continuous flux'
  //
  template<class Field>
  template<class Field_Vel>
  FluxValue<typename Advection_Flux<Field>::cfg> Advection_Flux<Field>::evaluate_continuos_flux(const FluxValue<cfg>& q,
                                                                                                const Field_Vel& vel,
                                                                                                const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < EquationData::dim);

    FluxValue<cfg> res;

    res = q*vel;

    return res;
  }


  // Implementation of the advection flux (left from right)
  //
  template<class Field>
  template<class Field_Vel>
  FluxValue<typename Advection_Flux<Field>::cfg> Advection_Flux<Field>::compute_discrete_flux_left_right(const FluxValue<cfg>& qL,
                                                                                                         const FluxValue<cfg>& qR,
                                                                                                         const Field_Vel& vel_L,
                                                                                                         const Field_Vel& vel_R,
                                                                                                         std::size_t curr_d) {
    FluxValue<cfg> res;

    //res = 0.5*(vel_L + vel_R) > 0.0 ? 0.0 : -0.5*(vel_L + vel_R)*(qL - qR); /*--- Working ---*/

    const auto lambda = std::max(std::abs(vel_L), std::abs(vel_R));

    res = 0.5*(vel_L*qL + vel_R*qR) - 0.5*lambda*(qR - qL) -
          (0.5*(vel_L + vel_R))*qL; /*--- Bassi-Rebay formulation ---*/

    return res;

  }


  // Implementation of the advection flux (right from left)
  //
  template<class Field>
  template<class Field_Vel>
  FluxValue<typename Advection_Flux<Field>::cfg> Advection_Flux<Field>::compute_discrete_flux_right_left(const FluxValue<cfg>& qL,
                                                                                                         const FluxValue<cfg>& qR,
                                                                                                         const Field_Vel& vel_L,
                                                                                                         const Field_Vel& vel_R,
                                                                                                         std::size_t curr_d) {
    FluxValue<cfg> res;

    //res = 0.5*(vel_L + vel_R) > 0.0 ? 0.5*(vel_L + vel_R)*(qR - qL) : 0.0; /*--- Working ---*/

    const auto lambda = std::max(std::abs(vel_L), std::abs(vel_R));

    res = -(0.5*(vel_L*qL + vel_R*qR) - 0.5*lambda*(qR - qL) -
            (0.5*(vel_L + vel_R))*qR); /*--- Bassi-Rebay formulation ---*/

    return res;
  }


  // Implement the contribution of the discrete flux for all the cells in the mesh.
  //
  template<class Field>
  template<class Field_Vect>
  auto Advection_Flux<Field>::make_flux(const Field_Vect& vel) {
    FluxDefinition<cfg> discrete_flux;

    // Perform the loop over each dimension to compute the flux contribution
    static_for<0, EquationData::dim>::apply(
      [&](auto integral_constant_d)
      {
        static constexpr int d = decltype(integral_constant_d)::value;

        // Compute now the "discrete" flux function
        discrete_flux[d].flux_function = [&](auto& cells, const Field& field)
                                              {
                                                const auto& left  = cells[0];
                                                const auto& right = cells[1];

                                                const auto& qL = field[left];
                                                const auto& qR = field[right];

                                                const auto& vel_L = vel[left];
                                                const auto& vel_R = vel[right];

                                                samurai::FluxValuePair<cfg> flux;
                                                flux[0] = compute_discrete_flux_left_right(qL, qR, vel_L, vel_R, d);
                                                flux[1] = compute_discrete_flux_right_left(qL, qR, vel_L, vel_R, d);

                                                return flux;
                                              };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }


  /**
    * Class to compute the flux between a left and right state (conservative form)
    */
  template<class Field>
  class Advection_Flux_Cons {
  public:
    // Definitions and sanity checks
    static constexpr std::size_t field_size        = Field::size;
    static_assert(Field::dim == EquationData::dim, "The spatial dimesions do not match");
    static constexpr std::size_t output_field_size = field_size;
    static constexpr std::size_t stencil_size      = 2;

    using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

    Advection_Flux_Cons() = default; // Default constructor

    template<class Field_Vel>
    FluxValue<cfg> evaluate_continuos_flux(const FluxValue<cfg>& q,
                                           const Field_Vel& vel,
                                           const std::size_t curr_d); // Evaluate the 'continuous' flux for the state q along direction curr_d

    template<class Field_Vel>
    FluxValue<cfg> compute_discrete_flux(const FluxValue<cfg>& qL,
                                         const FluxValue<cfg>& qR,
                                         const Field_Vel& vel_L,
                                         const Field_Vel& vel_R,
                                         const std::size_t curr_d); // advection flux along direction d

    template<class Field_Vect>
    auto make_flux(const Field_Vect& vel); // Compute the flux over all cells
  };


  // Evaluate the 'continuous flux'
  //
  template<class Field>
  template<class Field_Vel>
  FluxValue<typename Advection_Flux_Cons<Field>::cfg> Advection_Flux_Cons<Field>::evaluate_continuos_flux(const FluxValue<cfg>& q,
                                                                                                          const Field_Vel& vel,
                                                                                                          const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < EquationData::dim);

    FluxValue<cfg> res;

    res = q*vel;

    return res;
  }


  // Implementation of the advection flux
  //
  template<class Field>
  template<class Field_Vel>
  FluxValue<typename Advection_Flux_Cons<Field>::cfg> Advection_Flux_Cons<Field>::compute_discrete_flux(const FluxValue<cfg>& qL,
                                                                                                        const FluxValue<cfg>& qR,
                                                                                                        const Field_Vel& vel_L,
                                                                                                        const Field_Vel& vel_R,
                                                                                                        std::size_t curr_d) {
    FluxValue<cfg> res;

    const auto lambda = std::max(std::abs(vel_L), std::abs(vel_R));

    res = 0.5*(vel_L*qL + vel_R*qR) /*--- Centered contribution ---*/
        - 0.5*lambda*(qR - qL);     /*--- Stabilizing term ---*/

    return res;

  }


  // Implement the contribution of the discrete flux for all the cells in the mesh.
  //
  template<class Field>
  template<class Field_Vect>
  auto Advection_Flux_Cons<Field>::make_flux(const Field_Vect& vel) {
    FluxDefinition<cfg> discrete_flux;

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

                                                const auto& vel_L = vel[left];
                                                const auto& vel_R = vel[right];

                                                return compute_discrete_flux(qL, qR, vel_L, vel_R, d);
                                              };
      }
    );

    return make_flux_based_scheme(discrete_flux);
  }

} // end namespace samurai

#endif
