#include "RelaxedUT.h"
#include "UT.h"
#include <iostream>
#include <chrono>
#include <random>

#include "rmse.h"

using namespace RelaxedUnscentedTransformation;
using namespace Eigen;

const double Ts = 0.005;

// x = [ x y ω ϕc, Dx Dy   <- linear ones
//         phi, v, xc, yc, Dphi] <- nonlinear ones

// Total state-update dynamics as a large nonlinear function
VectorXd f(const VectorXd& x) {
	VectorXd out = VectorXd::Zero(11);
	out(0) = x(0) + Ts * x(7)*cos(x(6));
	out(1) = x(1) + Ts * x(7)*sin(x(6));
	out(2) = x(2);
	out(3) = x(3);
	out(4) = x(4) + Ts * x(7) * (cos(x(6) + x(10)) - cos(x(6)));
	out(5) = x(5) + Ts * x(7) * (sin(x(6) + x(10)) - sin(x(6)));
	out(6) = x(6) + Ts*x(2);
	out(7) = x(7);
	out(8) = x(8);
	out(9) = x(9);
	out(10) = x(10);
	return out;
}

// Total output-update characteristics as a large nonlinear function
VectorXd g(const VectorXd& x) {
	VectorXd out = VectorXd::Zero(6);
	out(0) = x(0) + x(8)*cos(x(6)) - x(9)*sin(x(6));
	out(1) = x(1) + x(9)*cos(x(6)) - x(8)*sin(x(6));
	out(2) = x(3) + x(6);
	out(3) = x(0) + x(4);
	out(4) = x(1) + x(5);
	out(5) = x(6) + x(10);
	return out;
}

// Nonlinear part of state-update dynamics - that depends on:
// xnl= [phi, v, xc, yc, Dphi]
VectorXd f2(const VectorXd& x) {
	VectorXd out = VectorXd::Zero(11);
	out(0) = Ts * x(7)*cos(x(6));
	out(1) = Ts * x(7)*sin(x(6));
	out(4) = Ts * x(7) * (cos(x(6) + x(10)) - cos(x(6)));
	out(5) = Ts * x(7) * (sin(x(6) + x(10)) - sin(x(6)));
	out(6) = x(6);
	out(7) = x(7);
	out(8) = x(8);
	out(9) = x(9);
	out(10) = x(10);
	return out;
}

// Nonlinear part of output-update characteristics - that depends on:
// xnl= [phi, v, xc, yc, Dphi]
VectorXd g2(const VectorXd& x) {
	VectorXd out = VectorXd::Zero(6);
	out(0) = x(8)*cos(x(6)) - x(9)*sin(x(6));
	out(1) = x(9)*cos(x(6)) - x(8)*sin(x(6));
	out(2) = x(6);
	out(5) = x(6) + x(10);
	return out;
}

#define MODIFIED

inline double kappa_abs(int N, int l) {
#ifndef MODIFIED
  switch (N) {
  case 2:
	if (l == 0)
	  return 3;
	break;
  case 3:
	switch (l) {
	case 0:
	  return 5.45;
	case 1:
	  return 0.55;
	}
  case 4:
	switch (l) {
	case 0:
	  return 6.894;
	case 1:
	  return 1.8658;
	case 2:
	  return 1.8658;
	}
  }
  throw std::runtime_error("kappa_abs(): Not allowed inputs");
  return 0;
}
#else
  switch (N) {
  case 2:
	if (l == 0)
	  return 3;
	break;
  case 3:
	switch (l) {
	case 0:
	  return 0.8911967313825753031460552474023;
	case 1:
	  return 6.0146721143623744947565745876628;
	}
  case 4:
	switch (l) {
	case 0:
	  return 0.9605;
	case 1:
	  return 2.9168;
	case 2:
	  return 8.3243;
	}
  }
  throw std::runtime_error("kappa_abs(): Not allowed inputs");
  return 0;
	}
#endif

int main (void) {
	// Clean simulation as ground truth
	for (auto Norder : { 2,3,4 }) {

	  std::vector<double> kappa;
	  for (int n = 0; n < Norder-1; n++)
		kappa.push_back(kappa_abs(Norder,n));
	  std::vector<double> alpha;
	  if (Norder == 2)
		alpha = { 1. };
	  if (Norder == 3)
		alpha = { 0.75, 1.25 };
	  if (Norder == 4)
		alpha = { 0.5, 1.,1.5 };
	  UTSettings settings;
	  settings.Norder = Norder;
	  settings.alpha = alpha;
	  settings.kappa = kappa;
	  std::vector<double> phi;
	  for (int n = 0; n < Norder - 1; n++)
		phi.push_back(0);
	  settings.initU(phi, 5);

	  double sumerr = RMSE(1, 250000, settings);

	  /*
	  for (int kk = 0; kk < 1; kk++) {
	  double sumerr = 0;

	  // Variable for simulation
	  double x_true = -1;
	  // Variables for filtering
	  double x_estimated = -0.5;
	  double Sx_estimated = 10;
	  for (int k = 1; k < 250000; k++) {
		//////////////// Simulation ////////////////
		// Simulation: update x
		double w = disturbance();
		x_true = f_scalar(x_true, w, k);
		// Simulation: compute z
		double n = noise();
		double z_measured = g_scalar(x_true, n);
		/////////////// Filtering ////////////////
		// Prediction step
		{
		  {
			Eigen::VectorXd xv(2);
			xv(0) = x_estimated;
			xv(1) = 0;
			Eigen::MatrixXd Sxw = Eigen::MatrixXd::Zero(2, 2);
			Sxw(0, 0) = Sx_estimated;
			Sxw(1, 1) = Sw;
			Eigen::VectorXd x_out;
			Eigen::MatrixXd Sxw_out;
			Eigen::MatrixXd Sxwoldnew_out;
			UT(xv, Sxw, [k](const Eigen::VectorXd& in)->Eigen::VectorXd {
			  Eigen::VectorXd out(1);
			  out(0) = f_scalar(in(0), in(1), k);
			  return out;
			  }, Norder, x_out, Sxw_out, Sxwoldnew_out);
			x_estimated = x_out(0);
			Sx_estimated = Sxw_out(0, 0);
		  }
		  //printf("Prediction Sx: %f\n", Sx_estimated);
		  double Sxz, Sz, z_estimated;
		  {
			Eigen::VectorXd xn(2);
			xn(0) = x_estimated;
			xn(1) = 0;

			Eigen::MatrixXd Sxn = Eigen::MatrixXd::Zero(2, 2);
			Sxn(0, 0) = Sx_estimated;
			Sxn(1, 1) = Sn;

			Eigen::VectorXd z_out;
			Eigen::MatrixXd Sz_out;
			Eigen::MatrixXd Szxw_out;

			UT(xn, Sxn, [](const Eigen::VectorXd& in)->Eigen::VectorXd {
			  Eigen::VectorXd out(1);
			  out(0) = g_scalar(in(0), in(1));
			  return out;
			  }, Norder, z_out, Sz_out, Szxw_out);
			z_estimated = z_out(0);
			Sz = Sz_out(0, 0);
			Sxz = Szxw_out(0, 0);
		  }

		  double K = Sxz / Sz;
		  x_estimated = x_estimated + K * (z_measured - z_estimated);
		  Sx_estimated = Sx_estimated - K * Sxz;
		  if (Sx_estimated < 0.01)
			Sx_estimated = 0.01;

		  sumerr += (x_estimated - x_true) * (x_estimated - x_true);

		  //printf("x_true: %f; x_estimated: %f, P_estimated: %f\n", x_true, x_estimated, Sx_estimated);
		}
	  }*/
	  printf("%d order: RMSE = %f\n", Norder, sumerr);
	}

  /*
	int Norder = 3;
	// Deriving matrix A
	MatrixXd A = MatrixXd::Zero(11, 6);
	for (int i = 0; i < 6; i++)
		A(i, i) = 1;
	A(6, 2) = Ts;
	// Deriving matrix C
	MatrixXd C = MatrixXd::Zero(6, 6);
	C(0, 0) = 1;
	C(1, 1) = 1;
	C(3, 0) = 1;
	C(4, 1) = 1;
	C(2, 3) = 1;
	C(3, 4) = 1;
	C(4, 5) = 1;
	// Inputvalues
	VectorXd x = VectorXd::Zero(11);
	x(0) = 0.3;
	x(1) = 0.4;
	x(2) = 0.1;
	x(3) = 0.05;
	x(4) = 0.01;
	x(5) = 0.005;
	x(6) = 0.5;
	x(7) = 1;
	x(8) = 0.001;
	x(9) = 0.002;
	x(10) = 0.01;

	Eigen::VectorXi il(6);
	for (int n = 0; n < 6; n++)
		il(n) = n;
	Eigen::VectorXi inl(5);
	for (int n = 0; n < 5; n++)
		inl(n) = n + 6;
	Eigen::MatrixXd F = Eigen::MatrixXd::Zero(0, 11);
	
	// Checking the function implementations
	std::cout << "The value of vector f(x_expected)\n";
	std::cout << " Totally nonlinear function: " << f(x).transpose() << std::endl;
	std::cout << " Partially linear function: " << (A * VectorSelect(x, il) + f2(x)).transpose() << std::endl;
	std::cout << "Seems OK.\n\n";
	
	// Checking computational time and precision of UT and RelaxedUT

	// covariance matrices
	MatrixXd S0 = MatrixXd::Zero(11,11);
	S0 << 0.0320, 0.0196, 0.0214, 0.0142, 0.0127, 0.0276, 0.0155, 0.0144, 0.0198, 0.0269, 0.0237,
		0.0196, 0.0481, 0.0237, 0.0256, 0.0189, 0.0369, 0.0197, 0.0235, 0.0249, 0.0338, 0.0300,
		0.0214, 0.0237, 0.0358, 0.0239, 0.0162, 0.0362, 0.0207, 0.0222, 0.0253, 0.0340, 0.0300,
		0.0142, 0.0256, 0.0239, 0.0668, 0.0154, 0.0323, 0.0250, 0.0234, 0.0285, 0.0294, 0.0245,
		0.0127, 0.0189, 0.0162, 0.0154, 0.0217, 0.0258, 0.0168, 0.0174, 0.0171, 0.0159, 0.0189,
		0.0276, 0.0369, 0.0362, 0.0323, 0.0258, 0.0570, 0.0290, 0.0322, 0.0327, 0.0412, 0.0360,
		0.0155, 0.0197, 0.0207, 0.0250, 0.0168, 0.0290, 0.0266, 0.0196, 0.0246, 0.0229, 0.0215,
		0.0144, 0.0235, 0.0222, 0.0234, 0.0174, 0.0322, 0.0196, 0.1889, 0.0215, 0.0316, 0.0265,
		0.0198, 0.0249, 0.0253, 0.0285, 0.0171, 0.0327, 0.0246, 0.0215, 0.0409, 0.0310, 0.0253,
		0.0269, 0.0338, 0.0340, 0.0294, 0.0159, 0.0412, 0.0229, 0.0316, 0.0310, 0.0657, 0.0446,
		0.0237, 0.0300, 0.0300, 0.0245, 0.0189, 0.0360, 0.0215, 0.0265, 0.0253, 0.0446, 0.0574;
	
	{
		long N_UT = (long)1e6;
		// Original UKF pred
		VectorXd original_x1, original_y;
		MatrixXd original_Sx1, original_Sx1x0, original_Sy, original_Syx1;
		auto start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++) {
			UT(x, S0, f, Norder, original_x1, original_Sx1, original_Sx1x0);
			UT(original_x1, original_Sx1, g, Norder, original_y, original_Sy, original_Syx1);
		}
		
		auto end = std::chrono::system_clock::now();
		auto dur_original = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		
		// Modified UKF pred
		VectorXd modified_x1, modified_y;
		MatrixXd modified_Sx1, modified_Sx1x0, modified_Syx, modified_Sy;
		start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++) {
			RelaxedUT(A, il, f2, Norder, F, inl, x, S0, modified_x1, modified_Sx1, modified_Sx1x0);
			RelaxedUT(C, il, g2, Norder, F, inl, modified_x1, modified_Sx1, modified_y, modified_Sy, modified_Syx);
		}
		end = std::chrono::system_clock::now();
		auto dur_relaxed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

		std::cout << "With the original UT:\n";
		std::cout << " the expected output: [" << original_y.transpose() << "]" << std::endl << std::endl;
		std::cout << " the expected output covariance matrix: [\n" << original_Sy << "]" << std::endl << std::endl;

		std::cout << "With the relaxed UT:\n";
		std::cout << " the expected output: [" << modified_y.transpose() << "]" << std::endl << std::endl;
		std::cout << " the expected output covariance matrix: [\n" << modified_Sy << "]" << std::endl << std::endl;

		printf("Average computational time of original UT : %f [us]\n", (double)dur_original / double(N_UT));
		printf("Average computational time of relaxed UT : %f [us]\n", (double)dur_relaxed / double(N_UT));
	}
	*/
	return 0;
}
