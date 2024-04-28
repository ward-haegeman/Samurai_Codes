#ifndef eos_hpp
#define eos_hpp

/**
  * Implementation of a generic lcass to handle the EOS. It has several
    pure virtual functions to be implementede for the specific EOS
  */
template<typename T = double>
class EOS {
public:
  static_assert(std::is_arithmetic_v<T>, "Template argument EOS not well suited for arithemtic operations");

  EOS() = default; // Default constructor

  EOS(const EOS&) = default; // Default copy-constructor

  virtual ~EOS() {} // Virtual destructor (it can be useful since we work thourgh the base class)

  virtual T pres_value(const T rho, const T e) const = 0; // Function to compute the pressure from the density and the internal energy

  virtual T rho_value(const T pres, const T e) const = 0; // Function to compute the density from the pressure and the internal energy

  virtual T e_value(const T rho, const T pres) const = 0; // Function to compute the internal energy from density and pressure

  virtual T c_value(const T rho, const T pres) const = 0; // Function to compute the speed of sound from density and pressure
};


/**
 * Implementation of the stiffened gas equation of state (SG-EOS)
 */
template<typename T = double>
class SG_EOS: public EOS<T> {
public:
  SG_EOS() = default; // Default constructor

  SG_EOS(const SG_EOS&) = default; // Default copy-constructor

  SG_EOS(const double gamma_, const double pi_infty_, const double q_infty_ = 0.0); // Constructor which accepts as arguments
                                                                                    // the isentropic exponent and the two parameters
                                                                                    // that characterize the fluid

  virtual T pres_value(const T rho, const T e) const override; // Function to compute the pressure from the density and the internal energy

  virtual T rho_value(const T pres, const T e) const override; // Function to compute the density from the pressure and the internal energy

  virtual T e_value(const T rho, const T pres) const override; // Function to compute the internal energy from density and pressure

  virtual T c_value(const T rho, const T pres) const override; // Function to compute the speed of sound from density and pressure

protected:
  const double gamma;    // Isentropic exponent
  const double pi_infty; // Pressure at 'infinite'
  const double q_infty;  // Internal energy at 'infinite'
};


// Implement the constructor
//
template<typename T>
SG_EOS<T>::SG_EOS(const double gamma_, const double pi_infty_, const double q_infty_):
  EOS<T>(), gamma(gamma_), pi_infty(pi_infty_), q_infty(q_infty_) {}


// Compute the pressure value from the density and the internal energy
//
template<typename T>
T SG_EOS<T>::pres_value(const T rho, const T e) const {
  return (gamma - 1.0)*rho*(e - q_infty) - gamma*pi_infty;
}


// Compute the density from the pressure and the internal energy
//
template<typename T>
T SG_EOS<T>::rho_value(const T pres, const T e) const {
  return (pres + gamma*pi_infty)/((gamma - 1.0)*(e - q_infty));
}


// Compute the internal energy from density and pressure
//
template<typename T>
T SG_EOS<T>::e_value(const T rho, const T pres) const {
  return (pres + gamma*pi_infty)/((gamma - 1.0)*rho) + q_infty;
}


// Compute the speed of sound from density and pressure
//
template<typename T>
T SG_EOS<T>::c_value(const T rho, const T pres) const {
  return std::sqrt(gamma*(pres + pi_infty)/rho);
}


#endif
