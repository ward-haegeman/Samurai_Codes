#ifndef ETAT_H_
#define ETAT_H_

using namespace std;

#include <sstream>

// fonction min et max
inline double max(double x1, double x2) 
{
	if (x1 < x2) return x2; else return x1;
}

inline double min(double x1, double x2) 
{
	if (x1 > x2) return x2; else return x1;
}


// La classe Etat
class Etat 
{	
	public: 
	
		// variables
		double al1, alrho1, alrhou1, alrhoE1, alrho2, alrhou2, alrhoE2;
	
		// Constructeur
		Etat() : al1(0.), alrho1(0.), alrhou1(0.), alrhoE1(0.), alrho2(0.), alrhou2(0.), alrhoE2(0.) {}
	
		Etat(double a, double b, double c, double d, double e, double f, double g) : 
			al1(a),	alrho1(b), alrhou1(c), alrhoE1(d), alrho2(e), alrhou2(f), alrhoE2(g) {}
	
		// Constructeur de copie
		Etat(const Etat &e) 
		{
			al1	= e.al1;
			alrho1	= e.alrho1;
			alrhou1	= e.alrhou1;
			alrhoE1= e.alrhoE1;
			alrho2	= e.alrho2;
			alrhou2	= e.alrhou2;
			alrhoE2= e.alrhoE2;
                }
	
		// Operateurs unaires
	
		// Fonctions membres
	
		// Operateurs binaires
		Etat operator=(const Etat &e) 
		{
			al1	= e.al1;
			alrho1	= e.alrho1;
			alrhou1	= e.alrhou1;
			alrhoE1= e.alrhoE1;
			alrho2	= e.alrho2;
			alrhou2	= e.alrhou2;
                        alrhoE2= e.alrhoE2;
			return *this;
		}
    
		Etat operator+(const Etat &e) 
		{
			return Etat
				(	al1	+ e.al1, 
					alrho1	+ e.alrho1,
					alrhou1	+ e.alrhou1,
					alrhoE1+ e.alrhoE1,
					alrho2	+ e.alrho2,
					alrhou2	+ e.alrhou2,
					alrhoE2+ e.alrhoE2
                                );
		}
		
		Etat operator-(const Etat &e) 
		{
			return Etat
				(	al1	- e.al1,
					alrho1	- e.alrho1,
					alrhou1	- e.alrhou1,
					alrhoE1- e.alrhoE1,
					alrho2	- e.alrho2,
					alrhou2	- e.alrhou2,
					alrhoE2- e.alrhoE2
                                );
		}
	
		// Operateurs binaires de multiplication et de division par un reel
		friend Etat operator*(const double &, const Etat &);
		friend Etat operator*(const Etat &, const double &);
		friend Etat operator/(const Etat &, const double &);
};

// Operateurs binaires de multiplication et de division par un reel
Etat operator*(const double &a, const Etat &e) 
{
	return Etat(		a*e.al1,
				a*e.alrho1,
				a*e.alrhou1,
				a*e.alrhoE1,
				a*e.alrho2,
				a*e.alrhou2,
				a*e.alrhoE2
                   );
}

Etat operator*(const Etat &e, const double &a) 
{
	return a*e;
}

Etat operator/(const Etat &e, const double &a) 
{
	return e*(1./a);
}

std::stringstream read_line(std::istream & in)
{
    std::string s;
    while (true)
    {
        std::getline(in, s);
        if (s[0] != '#')
        {
            return std::stringstream(s);
        }

    }
}


#endif
