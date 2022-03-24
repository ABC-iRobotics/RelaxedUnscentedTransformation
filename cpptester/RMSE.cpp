#include "RMSE.h"
#include "UT.h"
#include <iostream>
#include <chrono>
#include <random>

using namespace RelaxedUnscentedTransformation;
using namespace Eigen;

const double Ts = 0.005;

double f_scalar(double x, double w,int k) {
  return 0.5 * x + 25. * x / (1. + x * x) + 8 * cos(1.2 * double(k - 1)) + w;
}

double g_scalar(double x, double n) {
  return 0.05 * x * x + n;
}

#define Sw 0.035
#define Sn 0.01

double disturbance() {
  static std::default_random_engine generator;
  static std::normal_distribution<double> distribution(0, Sw);
  return distribution(generator);
}

double noise() {
  static std::default_random_engine generator;
  static std::normal_distribution<double> distribution(0, Sn);
  return distribution(generator);
}

double RMSE(int N, int K, const UTSettings& settings) {
  double out = 0;
  for (int n = 0; n < N; n++) {
	
	// Variable for simulation
	double x_true = -1;
	// Variables for filtering
	double x_estimated = -0.5;
	double Sx_estimated = 10;
	for (int k = 1; k < K; k++) {
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
			}, settings, x_out, Sxw_out, Sxwoldnew_out);
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
			}, settings, z_out, Sz_out, Szxw_out);
		  z_estimated = z_out(0);
		  Sz = Sz_out(0, 0);
		  Sxz = Szxw_out(0, 0);
		}
		
		double K = Sxz / Sz;
		x_estimated = x_estimated + K * (z_measured - z_estimated);
		Sx_estimated = Sx_estimated - K * Sxz;
		if (Sx_estimated < 0.01)
		  Sx_estimated = 0.01;

		out += (x_estimated - x_true) * (x_estimated - x_true);

		//printf("x_true: %f; x_estimated: %f, P_estimated: %f\n", x_true, x_estimated, Sx_estimated);
	  }
	}
  }
  return sqrt(out / double(K)) / double(N);
}
