#include "Eigen/Dense"
#include "UTMethods.h"
#include "IndexSelector.h"
#include <iostream>
#include <chrono>

using namespace UT;
using namespace Eigen;

const double Ts = 0.005;

// x = [ x y ω ϕc, Dx Dy   <- linear ones
//         phi, v, xc, yc, Dphi] <- nonlinear ones

// Total state-update dynamics as a large nonlinear function
struct TotallyNonlinearStateUpdate {
  static VectorXd f(const VectorXd& x) {
	VectorXd out = VectorXd::Zero(11);
	out(0) = x(0) + Ts * x(7) * cos(x(6));
	out(1) = x(1) + Ts * x(7) * sin(x(6));
	out(2) = x(2);
	out(3) = x(3);
	out(4) = x(4) + Ts * x(7) * (cos(x(6) + x(10)) - cos(x(6)));
	out(5) = x(5) + Ts * x(7) * (sin(x(6) + x(10)) - sin(x(6)));
	out(6) = x(6) + Ts * x(2);
	out(7) = x(7);
	out(8) = x(8);
	out(9) = x(9);
	out(10) = x(10);
	return out;
  }

  UT::ValWithCov UT(const UT::ValWithCov& x) {
	return UT::FullUT(f, x);
  }
};

struct TotallyNonlinearOutputUpdate {
  // Total output-update characteristics as a large nonlinear function
  static VectorXd g(const VectorXd& x) {
	VectorXd out = VectorXd::Zero(6);
	out(0) = x(0) + x(8) * cos(x(6)) - x(9) * sin(x(6));
	out(1) = x(1) + x(9) * cos(x(6)) - x(8) * sin(x(6));
	out(2) = x(3) + x(6);
	out(3) = x(0) + x(4);
	out(4) = x(1) + x(5);
	out(5) = x(6) + x(10);
	return out;
  }

  UT::ValWithCov UT(const UT::ValWithCov& x) {
	return UT::FullUT(g, x);
  }
};

struct RelaxedStateUpdate {
  Eigen::MatrixXd A, F;
  Eigen::VectorXi il, inl;

  RelaxedStateUpdate() {
	// Deriving matrix A
	A = Eigen::MatrixXd::Zero(11, 6);
	for (int i = 0; i < 6; i++)
	  A(i, i) = 1;
	A(6, 2) = Ts;
	// il, inl
	il = Eigen::VectorXi(6);
	for (int n = 0; n < 6; n++)
	  il(n) = n;
	inl = Eigen::VectorXi(5);
	for (int n = 0; n < 5; n++)
	  inl(n) = n + 6;
	F = Eigen::MatrixXd::Zero(0, 11);
  }

  // Nonlinear part of state-update dynamics - that depends on:
  // xnl= [phi, v, xc, yc, Dphi]
  static VectorXd f(const Eigen::VectorXd& x) {
	Eigen::VectorXd out = Eigen::VectorXd::Zero(11);
	out(0) = Ts * x(7) * cos(x(6));
	out(1) = Ts * x(7) * sin(x(6));
	out(4) = Ts * x(7) * (cos(x(6) + x(10)) - cos(x(6)));
	out(5) = Ts * x(7) * (sin(x(6) + x(10)) - sin(x(6)));
	out(6) = x(6);
	out(7) = x(7);
	out(8) = x(8);
	out(9) = x(9);
	out(10) = x(10);
	return out;
  }

  ValWithCov UT(const ValWithCov& x) {
	auto xdiffs = GenSigmaDifferences(x.Sy, inl);
	auto b0 = UTCore(x.y, xdiffs, f, UTOriginal(xdiffs.size()));
	auto b = LinearMappingOnb(b0, F);
	return MixedLinSources(x, b, il, A);
  }
};

struct RelaxedOutputUpdate {
  Eigen::MatrixXd A, F;
  Eigen::VectorXi il, inl;

  RelaxedOutputUpdate() {
	// Deriving matrix C
	Eigen::MatrixXd C = Eigen::MatrixXd::Zero(6, 6);
	C(0, 0) = 1;
	C(1, 1) = 1;
	C(3, 0) = 1;
	C(4, 1) = 1;
	C(2, 3) = 1;
	C(3, 4) = 1;
	C(4, 5) = 1;
	// il, inl
	il = Eigen::VectorXi(6);
	for (int n = 0; n < 6; n++)
	  il(n) = n;
	inl = Eigen::VectorXi(5);
	for (int n = 0; n < 5; n++)
	  inl(n) = n + 6;
	F = Eigen::MatrixXd::Zero(0, 11);
  }

  // Nonlinear part of output-update characteristics - that depends on:
  // xnl= [phi, v, xc, yc, Dphi]
  static VectorXd g(const VectorXd& x) {
	VectorXd out = VectorXd::Zero(6);
	out(0) = x(8) * cos(x(6)) - x(9) * sin(x(6));
	out(1) = x(9) * cos(x(6)) - x(8) * sin(x(6));
	out(2) = x(6);
	out(5) = x(6) + x(10);
	return out;
  }
  
  ValWithCov UT(const ValWithCov& x) {
	auto xdiffs = GenSigmaDifferences(x.Sy, inl);
	auto b0 = UTCore(x.y, xdiffs, g, UTOriginal(xdiffs.size()));
	auto b = LinearMappingOnb(b0, F);
	return MixedLinSources(x, b, il, A);
  }
};

int main (void) {
	// Inputvalues
	Eigen::VectorXd x_ = Eigen::VectorXd::Zero(11);
	x_(0) = 0.3;
	x_(1) = 0.4;
	x_(2) = 0.1;
	x_(3) = 0.05;
	x_(4) = 0.01;
	x_(5) = 0.005;
	x_(6) = 0.5;
	x_(7) = 1;
	x_(8) = 0.001;
	x_(9) = 0.002;
	x_(10) = 0.01;
	// covariance matrices
	MatrixXd S0 = MatrixXd::Zero(11, 11);
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
	ValWithCov x(x_, S0);

	TotallyNonlinearStateUpdate stateUpdate0;
	TotallyNonlinearOutputUpdate outputUpdate0;
	RelaxedStateUpdate stateUpdate1;
	RelaxedOutputUpdate outputUpdate1;
	
	// Checking the function implementations
	std::cout << "The value of vector f(x_expected)\n";
	std::cout << " Totally nonlinear function: " << stateUpdate0.f(x_).transpose() << std::endl;
	std::cout << " Partially linear function: " << (stateUpdate1.A * VectorSelect(x_, stateUpdate1.il)
	  + stateUpdate1.f(x_)).transpose() << std::endl;
	std::cout << "Seems OK.\n\n";
	
	// Checking computational time and precision of UT and RelaxedUT
	{
		long N_UT = (long)1e6;
		// Original UKF pred
		ValWithCov x1_original, y_original;
		auto start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++) {
		  x1_original = stateUpdate0.UT(x);
		  y_original = outputUpdate0.UT(x1_original);
		}
		auto end = std::chrono::system_clock::now();
		auto dur_original = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		
		// Modified UKF pred
		ValWithCov x1_relaxed, y_relaxed;
		start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++) {
		  x1_relaxed = stateUpdate1.UT(x);
		  y_relaxed = outputUpdate1.UT(x1_relaxed);
		}
		end = std::chrono::system_clock::now();
		auto dur_relaxed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

		std::cout << "With the original UT:\n";
		std::cout << " the expected output: [" << x1_original.y.transpose() << "]" << std::endl << std::endl;
		std::cout << " the expected output covariance matrix: [\n" << x1_original.Sy << "]" << std::endl << std::endl;

		std::cout << "With the relaxed UT:\n";
		std::cout << " the expected output: [" << x1_relaxed.y.transpose() << "]" << std::endl << std::endl;
		std::cout << " the expected output covariance matrix: [\n" << x1_relaxed.Sy << "]" << std::endl << std::endl;

		printf("Average computational time of original UT : %f [us]\n", (double)dur_original / double(N_UT));
		printf("Average computational time of relaxed UT : %f [us]\n", (double)dur_relaxed / double(N_UT));
	}
	return 0;
}
