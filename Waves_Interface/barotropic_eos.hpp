#ifndef barotropic_eos_hpp
#define barotropic_eos_hpp

/**
 * Generic interface for a barotropic EOS
 */
template<typename T = double>
class BarotropicEOS {
public:
  BarotropicEOS() = default; // Default constructor

  BarotropicEOS(const BarotropicEOS&) = default; // Default copy-constructor

  virtual ~BarotropicEOS() = default;

  virtual T pres_value(const T& rho) const = 0; // Function to actually compute the pressure from the density

  virtual T c_value(const T& rho) const = 0; // Function to compute the speed of sound

  virtual T rho_value(const T& pres) const = 0; // Function to compute the density from the pressure
};


/**
 * Implementation of a linearized barotropic EOS
 */
template<typename T = double>
class LinearizedBarotropicEOS: public BarotropicEOS<T> {
public:
  LinearizedBarotropicEOS() = default; // Default constructor

  LinearizedBarotropicEOS(const LinearizedBarotropicEOS&) = default; // Default copy-constructor

  LinearizedBarotropicEOS(const double p0_, const double rho0_, const double c0_); // Constructor which accepts as arguments
                                                                                   // reference pressure, density and speed of sound

  virtual T pres_value(const T& rho) const override; // Function to actually compute the pressure from the density

  virtual T c_value(const T& rho) const override; // Function to compute the speed of sound

  virtual T rho_value(const T& pres) const override; // Function to compute the density from the pressure

  inline double get_c0() const; // Get the speed of sound

  inline double get_p0() const; // Get the reference pressure

  inline double get_rho0() const; // Get the reference density

  inline void set_c0(const double c0_); // Set the speed of sound

  inline void set_p0(const double p0_); // Set the reference pressure

  inline void set_rho0(const double rho0_); // Set the reference density

private:
  double p0;   // Reference pressure
  double rho0; // Reference density
  double c0;   // Speed of sound
};

// Implement the constructor
//
template<typename T>
LinearizedBarotropicEOS<T>::LinearizedBarotropicEOS(const double p0_, const double rho0_, const double c0_):
  BarotropicEOS<T>(), p0(p0_), rho0(rho0_), c0(c0_) {}

// Implement the pressure value from the density
//
template<typename T>
T LinearizedBarotropicEOS<T>::pres_value(const T& rho) const {
  if(std::isnan(rho)) {
    return nan("");
  }

  return p0 + c0*c0*(rho - rho0);
}

// Implement the speed of sound from the density
//
template<typename T>
T LinearizedBarotropicEOS<T>::c_value(const T& rho) const {
  (void) rho;

  return c0;
}

// Implement the density from the pressure
//
template<typename T>
T LinearizedBarotropicEOS<T>::rho_value(const T& pres) const {
  if(std::isnan(pres)) {
    return nan("");
  }

  return (pres - p0)/(c0*c0) + rho0;
}

// Implement the getter of the speed of sound
//
template<typename T>
inline double LinearizedBarotropicEOS<T>::get_c0() const {
  return c0;
}

// Implement the getter of the reference pressure
//
template<typename T>
inline double LinearizedBarotropicEOS<T>::get_p0() const {
  return p0;
}

// Implement the getter of the reference density
//
template<typename T>
inline double LinearizedBarotropicEOS<T>::get_rho0() const {
  return rho0;
}

// Implement the setter for the speed of sound
//
template<typename T>
inline void LinearizedBarotropicEOS<T>::set_c0(const double c0_) {
  c0 = c0_;
}

// Implement the setter of the reference pressure
//
template<typename T>
inline void LinearizedBarotropicEOS<T>::set_p0(const double p0_) {
  p0 = p0_;
}

// Implement the setter of the reference density
//
template<typename T>
inline void LinearizedBarotropicEOS<T>::set_rho0(const double rho0_) {
  rho0 = rho0_;
}

#endif
