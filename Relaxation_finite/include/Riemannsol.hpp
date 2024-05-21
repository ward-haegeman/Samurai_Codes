double M0 (double nu, double Me)
{
	return 4./(nu+1.)*Me/(1.+Me*Me)/(1.+sqrt(fabs(1.-4.*nu/(nu+1.)/(nu+1.)*4.*Me*Me/(1.+Me*Me)/(1.+Me*Me))));
}

// Derivee de M0 par rapport a Me, pour le Newton
double dMM0 (double nu, double Me)
{
        double w = (1.-Me)/(1.+Me);
        return 4./(nu+1.)*w/(1.+w*w)/(1.+w*w)*(1.+w)*(1.+w)/(1.-4.*nu/(nu+1.)/(nu+1.)*(1.-w*w)*(1.-w*w)/(1.+w*w)/(1.+w*w)+sqrt(fabs(1.-4.*nu/(nu+1.)/(nu+1.)*(1.-w*w)*(1.-w*w)/(1.+w*w)/(1.+w*w))));
}

double psi(double ue, double a, double alphaL, double alphaR, double ud, double tLd, double tRd)
{
	if (ue <= ud)
        {
            return a*(alphaL+alphaR)*(ue-ud)+2.*a*a*alphaL*tLd*M0(alphaL/alphaR,(ud-ue)/(a*tLd));
        }
	else
        {
                return -psi(-ue, a, alphaR, alphaL, -ud, tRd, tLd);
        }
}

// Derivee de psi par rapport a ue, pour le Newton
double duepsi(double ue, double a, double alphaL, double alphaR, double ud, double tLd, double tRd)
{
	if (ue <= ud)
		return a*(alphaL+alphaR)-2.*a*alphaL*dMM0(alphaL/alphaR,(ud-ue)/(a*tLd));

	else
		return a*(alphaL+alphaR)-2.*a*alphaR*dMM0(alphaR/alphaL,(ud-ue)/(a*tRd));
}


double Psi(double ue, double a1, double alpha1L, double alpha1R, double u1d, double t1Ld, double t1Rd, double a2, double alpha2L, double alpha2R, double u2d, double t2Ld, double t2Rd)
{
	return  a1*(alpha1L+alpha1R)*(ue-u1d)+ psi(ue, a2, alpha2L, alpha2R, u2d, t2Ld, t2Rd);
}

// Derivee de Psi par rapport a ue, pour le Newton
double duePsi(double ue, double a1, double alpha1L, double alpha1R, double u1d, double t1Ld, double t1Rd, double a2, double alpha2L, double alpha2R, double u2d, double t2Ld, double t2Rd)
{
        return a1*(alpha1L+alpha1R)+duepsi(ue, a2, alpha2L, alpha2R, u2d, t2Ld, t2Rd);
}


double Newton (double cible, double a1, double alpha1L, double alpha1R, double u1d, double t1Ld, double t1Rd, double a2, double alpha2L, double alpha2R, double u2d, double t2Ld, double t2Rd, double eps, int &compt)
{
        double	xL, xR, ue;

	compt = 0;

	/*double umax = bornsup;
	double umin = borninf;*/

	if (alpha1L==alpha1R) return u1d;

	else
	{
		xL = max(u1d-a1*t1Ld,u2d-a2*t2Ld); xR = min(u1d+a1*t1Rd,u2d+a2*t2Rd);
                ue= (xL+xR)/2.;

		double du = -(Psi(ue, a1, alpha1L, alpha1R, u1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, u2d, t2Ld, t2Rd)-cible)/
								 duePsi(ue, a1, alpha1L, alpha1R, u1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, u2d, t2Ld, t2Rd);

		while ((fabs(Psi(ue, a1, alpha1L, alpha1R, u1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, u2d, t2Ld, t2Rd)-cible)>eps) && (compt<50) && (fabs(du) > eps))
		{
										/*if(du > 0.0) {
												du = max(du, 0.9*(umax - ue));
												umin = ue;
											}
										else if(du < 0.0) {
											du = max(du, 0.9*(umin - ue));
											umax = ue;
										}*/
										ue = ue + du;

                    compt++;

										du = -(Psi(ue, a1, alpha1L, alpha1R, u1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, u2d, t2Ld, t2Rd)-cible)/
													duePsi(ue, a1, alpha1L, alpha1R, u1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, u2d, t2Ld, t2Rd);

		}
		if (compt==50)
		{
		      cout << "Newton pas convergé..." << endl;
		      exit(0);
		}
		return	ue;
	}
}


double Dichotomie (double cible, double a1, double alpha1L, double alpha1R, double u1d, double t1Ld, double t1Rd, double a2, double alpha2L, double alpha2R, double u2d, double t2Ld, double t2Rd, double eps, int &compt)
{
	double	xL, xR, ue;

	compt = 0;

	if (alpha1L==alpha1R) return u1d;
	else
	{
		xL = max(u1d-a1*t1Ld,u2d-a2*t2Ld); xR = min(u1d+a1*t1Rd,u2d+a2*t2Rd);
                //xL=0.4; xR=0.6;
		ue= (xL+xR)/2.;

		while ((fabs(Psi(ue, a1, alpha1L, alpha1R, u1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, u2d, t2Ld, t2Rd)-cible)>eps)&&(compt<1000))
		{
			ue= (xL+xR)/2.;
			if ( (Psi(ue, a1, alpha1L, alpha1R, u1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, u2d, t2Ld, t2Rd)-cible)*(Psi(xL, a1, alpha1L, alpha1R, u1d, t1Ld, t1Rd, a2, alpha2L, alpha2R, u2d, t2Ld, t2Rd)-cible) >0.)
			{
				xL = ue;
			}
	  		else
			{
				xR = ue;
			}
	  		compt++;
		}
		if (compt==1000)
		{
			cout << "Dichotomie pas convergée..." << endl;
			exit(0);
	}
		}
	return	ue;
}

// Solveur de Riemann pour la phase associee a la vitesse d'interface
void RiemannsolP1 (double ksi,
		   double alphaL, double alphaR, double tauL, double tauR, double wL, double wR, double pL, double pR, double EL, double ER,
		   double a, double ue, double &alpham, double &taum, double &wm, double &pim, double &Em, double &alphap, double &taup, double &wp, double &pip, double &Ep)
{
	if (ksi < wL-a*tauL)
	{
		alpham = alphaL;
		taum	= tauL;
		wm	= wL;
		pim	= pL;
                Em      = EL;

		alphap = alphaL;
		taup	= tauL;
		wp	= wL;
		pip	= pL;
                Ep      = EL;
	}
	else
	{
		if (ksi == wL-a*tauL)
		{
			alpham = alphaL;
			taum	= tauL;
			wm	= wL;
			pim	= pL;
                        Em      = EL;

			alphap = alphaL;
			taup	= tauL + 1./a*(ue-wL);
			wp	= ue;
			pip	= pL + a*(wL-ue);
                        Ep	= EL -1./a*(pip*wp-pL*wL);
		}
		else
		{
			if (ksi > wL-a*tauL && ksi < ue)
			{
				alpham = alphaL;
				taum	= tauL + 1./a*(ue-wL);
				wm	= ue;
				pim	= pL + a*(wL-ue);
                                Em	= EL -1./a*(pim*wm-pL*wL);

				alphap = alphaL;
				taup	= tauL + 1./a*(ue-wL);
				wp	= ue;
				pip	= pL + a*(wL-ue);
                                Ep	= EL -1./a*(pip*wp-pL*wL);
			}
			else
			{
				if (ksi==ue)
				{
					alpham = alphaL;
					taum	= tauL + 1./a*(ue-wL);
					wm	= ue;
					pim	= pL + a*(wL-ue);
                                        Em	= EL -1./a*(pim*wm-pL*wL);

					alphap = alphaR;
					taup	= tauR - 1./a*(ue-wR);
					wp	= ue;
					pip	= pR - a*(wR-ue);
                                        Ep	= ER +1./a*(pip*wp-pR*wR);
				}
				else
				{
					if (ksi > ue && ksi < wR+a*tauR)
					{
						alpham = alphaR;
						taum	= tauR - 1./a*(ue-wR);
						wm	= ue;
						pim	= pR - a*(wR-ue);
                                                Em	= ER +1./a*(pim*wm-pR*wR);

						alphap = alphaR;
						taup	= tauR - 1./a*(ue-wR);
						wp	= ue;
						pip	= pR - a*(wR-ue);
                                                Ep	= ER +1./a*(pip*wp-pR*wR);
					}
					else
					{
						if (ksi==wR+a*tauR)
						{
							alpham = alphaR;
							taum	= tauR - 1./a*(ue-wR);
							wm	= ue;
							pim	= pR - a*(wR-ue);
                                                        Em	= ER +1./a*(pim*wm-pR*wR);

							alphap = alphaR;
							taup	= tauR;
							wp	= wR;
							pip	= pR;
                                                        Ep	= ER;
						}
						else
						{
							alpham = alphaR;
							taum	= tauR;
							wm	= wR;
							pim	= pR;
                                                        Em	= ER;

							alphap = alphaR;
							taup	= tauR;
							wp	= wR;
							pip	= pR;
                                                        Ep	= ER;
						}
					}
				}
			}
		}
	}
}


// Solveur de Riemann pour la phase associee a la pression d'interface, dans le referentiel de u_1
void Riemannsoldec (double ksi,
		      double alphaL, double alphaR, double tauL, double tauR, double wL, double wR, double pL, double pR, double EL, double ER, double wd, double pid, double taudL, double taudR,
		      double a, double &alpham, double &taum, double &wm, double &pim, double &Em, double &alphap, double &taup, double &wp, double &pip, double &Ep, int &dissip)
{
 	double nu	= alphaL/alphaR;
	double ML	= wL/a/tauL;
	double MdL	= wd/a/taudL;

	double M;
	double Mzero;
	double mu	= 0.9;
	double nuc;
	double t 	= taudR/taudL;

	dissip 		= 0;

	if (wd > 0.)
	{
		if (ML <  1.)  //CONFIGURATION <1,2> subsonique.
		{   // CALCUL DE M QUI PARAMETRISE TTE LA SOLUTION


			Mzero= 4./(nu+1.)*MdL/(1.+MdL*MdL)/(1.+sqrt(fabs(1.-4.*nu/(nu+1.)/(nu+1.)*4.*MdL*MdL/(1.+MdL*MdL)/(1.+MdL*MdL))));


			if (mu*taudR <= taudR+taudL*(MdL+nu*Mzero)/(1.+nu*Mzero))	M = Mzero;
                        else
			{
                            // Cas ou il faut dissiper de l'energie
			    M 	= 1./nu*(MdL+t*(1-mu))/(1.-t*(1.-mu));
                            dissip	= 1;
			}
		}

		if (ksi < wL-a*tauL)
		{
			alpham = alphaL;
			taum	= tauL;
			wm	= wL;
			pim	= pL;
                        Em	= EL;

			alphap = alphaL;
			taup	= tauL;
			wp	= wL;
			pip	= pL;
                        Ep	= EL;
		}
		else
		{
			if (ksi == wL-a*tauL)
			{
				alpham = alphaL;
				taum	= tauL;
				wm	= wL;
				pim	= pL;
                                Em	= EL;

				alphap = alphaL;
				taup	= taudL*(1.-MdL)/(1.-M);
				wp	= a*M*taup;
				pip	= pL + a*(wL-wp);
                                Ep	= EL -1./a*(pip*wp-pL*wL);
			}
			else
			{
				if (ksi > wL-a*tauL && ksi < 0.)
				{
					alpham = alphaL;
					taum	= taudL*(1.-MdL)/(1.-M);
					wm	= a*M*taum;
					pim	= pL + a*(wL-wm);
                                        Em	= EL-1./a*(pim*wm-pL*wL);

				  	alphap = alphaL;
					taup	= taudL*(1.-MdL)/(1.-M);
					wp	= a*M*taup;
					pip	= pL + a*(wL-wp);
                                        Ep	= EL-1./a*(pip*wp-pL*wL);
				}
				else
				{
					if (ksi == 0.)
					{
						alpham = alphaL;
						taum	= taudL*(1.-MdL)/(1.-M);
						wm	= a*M*taum;
						pim	= pL + a*(wL-wm);
                                                Em	= EL-1./a*(pim*wm-pL*wL);

						alphap = alphaR;
						taup	= taudL*(1.+MdL)/(1.+nu*M);
						wp	= nu*a*M*taup;
						pip	= pL + a*a*(tauL-taup);
                                                Ep	= Em-1./(nu*a)*(pip*nu*a*taup-nu*pim*a*taum);

					}
					else
					{
						if (ksi > 0. && ksi < nu*a*M*taudL*(1.+MdL)/(1.+nu*M))
						{
							///// CE BLOC NE SERT QU AU CALCUL DE E1m ET E1p
						  	alpham = alphaL;
                                                        taum	= taudL*(1.-MdL)/(1.-M);
                                                        wm	= a*M*taum;
                                                        pim	= pL + a*(wL-wm);
                                                        Em	= EL-1./a*(pim*wm-pL*wL);

                                                        alphap = alphaR;
                                                        taup	= taudL*(1.+MdL)/(1.+nu*M);
                                                        wp	= nu*a*M*taup;
                                                        pip	= pL + a*a*(tauL-taup);
                                                        Ep	= Em-1./(nu*a)*(pip*nu*a*taup-nu*pim*a*taum);
						  	/////// FIN DU BLOC ///////////////////////


                                                        alpham = alphaR;
							taum	= taudL*(1.+MdL)/(1.+nu*M);
							wm	= nu*a*M*taum;
							pim	= pL + a*a*(tauL-taum);
                                                        Em      = Ep;

						  	alphap = alphaR;
							taup	= taudL*(1.+MdL)/(1.+nu*M);
							wp	= nu*a*M*taup;
							pip	= pL + a*a*(tauL-taup);
						}
						else
						{
							if (ksi == nu*a*M*taudL*(1.+MdL)/(1.+nu*M))
							{

                                                                ///// CE BLOC NE SERT QU AU CALCUL DE E1m
                                                                alpham = alphaL;
                                                                taum	= taudL*(1.-MdL)/(1.-M);
                                                                wm	= a*M*taum;
                                                                pim	= pL + a*(wL-wm);
                                                                Em	= EL-1./a*(pim*wm-pL*wL);

                                                                alphap = alphaR;
                                                                taup	= taudL*(1.+MdL)/(1.+nu*M);
                                                                wp	= nu*a*M*taup;
                                                                pip	= pL + a*a*(tauL-taup);
                                                                Ep	= Em-1./(nu*a)*(pip*nu*a*taup-nu*pim*a*taum);
						  	/////// FIN DU BLOC ///////////////////////

                                                                alpham = alphaR;
								taum	= taudL*(1.+MdL)/(1.+nu*M);
								wm	= nu*a*M*taum;
								pim	= pL + a*a*(tauL-taum);
                                                                Em      = Ep;

								alphap = alphaR;
								taup	= taudR+taudL*(MdL-nu*M)/(1.+nu*M);
								wp	= nu*a*M*taudL*(1.+MdL)/(1.+nu*M);
								pip	= pR - a*(wR-wp);
                                                                Ep      = ER-1./a*(pR*wR-pip*wp);
							}
							else
							{
								if (ksi > nu*a*M*taudL*(1.+MdL)/(1.+nu*M) && ksi < wR+a*tauR)
								{
									alpham = alphaR;
									taum	= taudR+taudL*(MdL-nu*M)/(1.+nu*M);
									wm	= nu*a*M*taudL*(1.+MdL)/(1.+nu*M);
									pim	= pR - a*(wR-wm);
                                                                        Em      = ER-1./a*(pR*wR-pim*wm);

									alphap = alphaR;
									taup	= taudR+taudL*(MdL-nu*M)/(1.+nu*M);
									wp	= nu*a*M*taudL*(1.+MdL)/(1.+nu*M);
									pip	= pR - a*(wR-wp);
                                                                        Ep      = ER-1./a*(pR*wR-pip*wp);
								}
								else
								{
									if ( ksi == wR+a*tauR)
									{
										alpham = alphaR;
										taum	= taudR+taudL*(MdL-nu*M)/(1.+nu*M);
										wm	= nu*a*M*taudL*(1.+MdL)/(1.+nu*M);
										pim	= pR - a*(wR-wm);
                                                                                Em      = ER-1./a*(pR*wR-pim*wm);

										alphap = alphaR;
										taup	= tauR;
										wp	= wR;
										pip	= pR;
                                                                                Ep      = ER;
									}
									else
									{
										alpham = alphaR;
										taum	= tauR;
										wm	= wR;
										pim	= pR;
                                                                                Em      = ER;

										alphap = alphaR;
										taup	= tauR;
										wp	= wR;
										pip	= pR;
                                                                                Ep      = ER;
									}
								}
							}
						}
					}

				}
			}
		}
	}

	else // CONFIGURATIONS SYMETRIQUES
	{
		if (wd < 0.)
                {
                        Riemannsoldec	(-ksi,
                                        alphaR, alphaL, tauR, tauL, -wR, -wL, pR, pL, ER, EL, -wd, pid, taudR, taudL,
                                        a, alphap, taup, wp, pip, Ep, alpham, taum, wm, pim, Em, dissip);
                        wm = wm*(-1.);
                        wp = wp*(-1.);
                }

                else //CONFIGURATION wd=0
                {
                        if (ksi < wL-a*tauL)
			{
				alpham = alphaL;
				taum	= tauL;
				wm	= wL;
				pim	= pL;
				Em	= EL;

				alphap = alphaL;
				taup	= tauL;
				wp	= wL;
				pip	= pL;
				Ep	= EL;
			}
			else
			{
				if (ksi == wL-a*tauL)
				{
					alpham = alphaL;
					taum	= tauL;
					wm	= wL;
					pim	= pL;
					Em	= EL;

					alphap = alphaL;
					taup	= taudL;
					wp	= 0.;
					pip	= pL + a*(wL-wp);
					Ep	= EL-1./a*(pip*wp-pL*wL);
				}
				else
				{
					if (ksi > wL-a*tauL && ksi < 0.)
					{
						alpham = alphaL;
						taum	= taudL;
						wm	= 0.;
						pim	= pL + a*(wL-wm);
						Em	= EL-1./a*(pim*wm-pL*wL);

						alphap = alphaL;
						taup	= taudL;
						wp	= 0.;
						pip	= pL + a*(wL-wp);
						Ep	= EL-1./a*(pip*wp-pL*wL);
					}
					else
					{
						if (ksi == 0.)
						{
							alpham = alphaL;
							taum	= taudL;
							wm	= 0.;
							pim	= pL + a*(wL-wm);
							Em	= EL-1./a*(pim*wm-pL*wL);

							alphap = alphaR;
							taup	= taudR;
							wp	= 0.;
							pip	= pR - a*(wR-wp);
							Ep	= ER+1./a*(pip*wp-pR*wR);
						}
						else
						{
							if (ksi > 0. && ksi < wR+a*tauR)
							{
								alpham = alphaR;
								taum	= taudR;
								wm	= 0.;
								pim	= pR - a*(wR-wm);
								Em	= ER+1./a*(pim*wm-pR*wR);

								alphap = alphaR;
								taup	= taudR;
								wp	= 0.;
								pip	= pR - a*(wR-wp);
								Ep	= ER+1./a*(pip*wp-pR*wR);
							}
							else
							{
								if (ksi == wR+a*tauR)
								{
									alpham = alphaR;
									taum	= taudR;
									wm	= 0.;
									pim	= pR - a*(wR-wm);
									Em	= ER+1./a*(pim*wm-pR*wR);

									alphap = alphaR;
									taup	= tauR;
									wp	= wR;
									pip	= pR;
									Ep	= ER;
								}
								else
								{
									alpham = alphaR;
									taum	= tauR;
									wm	= wR;
									pim	= pR;
									Em	= ER;

									alphap = alphaR;
									taup	= tauR;
									wp	= wR;
									pip	= pR;
									Ep	= ER;
								}
							}
						}
					}
				}
			}
		}


	}
}
