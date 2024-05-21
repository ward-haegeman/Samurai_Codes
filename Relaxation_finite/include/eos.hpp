#ifndef EOS_H_
#define EOS_H_

#include "etat.hpp"


// Loi de pression phase 1

double Gamma1 = 3.0;
double pp1 = 100.0;


inline double p1(double r1, double e1)
{
	return (Gamma1-1.)*r1*e1-Gamma1*pp1;
}


inline double P1(Etat e)
{
  double r1 = e.alrho1/e.al1;
  double e1 = (e.alrhoE1-e.alrhou1*e.alrhou1/2./e.alrho1)/e.alrho1;
  return p1(r1,e1);
}

// energie interne specifique phase 1
inline double e1(double r1, double p1)
{
	return (p1+Gamma1*pp1)/(Gamma1-1.)/r1;
}


// vitesse du son phase 1

inline double cc1(double r1, double e1)
{
	return sqrt(Gamma1*(p1(r1,e1)+pp1)/r1);
}

inline double c1(Etat e)
{
  double r1 = e.alrho1/e.al1;
  double e1 = (e.alrhoE1-e.alrhou1*e.alrhou1/2./e.alrho1)/e.alrho1;
  return cc1(r1,e1);
}

// Loi de pression phase 2

double Gamma2 = 1.4;
double pp2 = 0.0;


inline double p2(double r2, double e2)
{
	return (Gamma2-1.)*r2*e2-Gamma2*pp2;
}

inline double P2(Etat e)
{
  double r2 = e.alrho2/(1.-e.al1);
  double e2 = (e.alrhoE2-e.alrhou2*e.alrhou2/2./e.alrho2)/e.alrho2;
  return p2(r2,e2);
}

// energie interne specifique phase 2
inline double e2(double r2, double p2)
{
	return (p2+Gamma2*pp2)/(Gamma2-1.)/r2;
}

// vitesse du son phase 2

inline double cc2(double r2, double e2)
{
	return sqrt(Gamma2*(p2(r2,e2)+pp2)/r2);
}

inline double c2(Etat e)
{
  double r2 = e.alrho2/(1.-e.al1);
  double e2 = (e.alrhoE2-e.alrhou2*e.alrhou2/2./e.alrho2)/e.alrho2;
  return cc2(r2,e2);
}



#endif
