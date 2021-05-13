#include "common/unity.h"
void setUp() {}
void tearDown() {}

#include "defs.h"
#include "StatisticValue.h"
#include "PartialCholevski.h"
#include <iostream>

using namespace SF;
using namespace Eigen;

const double Ts = 0.005;

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

// xnl= phi, v, xc, yc, Dphi
VectorXd f2(const VectorXd& xnl) {
	VectorXd out = VectorXd::Zero(11);
	out(0) = Ts * xnl(1)*cos(xnl(0));
	out(1) = Ts * xnl(1)*sin(xnl(0));
	out(4) = Ts * xnl(1) * (cos(xnl(0) + xnl(4)) - cos(xnl(0)));
	out(5) = Ts * xnl(1) * (sin(xnl(0) + xnl(4)) - sin(xnl(0)));
	out(6) = xnl(0);
	out(7) = xnl(1);
	out(8) = xnl(2);
	out(9) = xnl(3);
	out(10) = xnl(4);
	return out;
}

VectorXd g2(const VectorXd& xnl) {
	VectorXd out = VectorXd::Zero(6);
	out(0) = xnl(2)*cos(xnl(0)) - xnl(3)*sin(xnl(0));
	out(1) = xnl(3)*cos(xnl(0)) - xnl(2)*sin(xnl(0));
	out(2) = xnl(0);
	out(5) = xnl(0) + xnl(4);
	return out;
}

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

// x = [ xl:R^(n-m)  xnl:R^m ]
void UT(const VectorXd& x0, const MatrixXd& S0, int m, VectorXd (*fin)(const VectorXd&),
	VectorXd& z, MatrixXd& Sz, MatrixXd& Szx) {
	int n = x0.size();
	auto xl = x0.block(0, 0, n - m, 1);
	auto xnl = x0.block(n - m, 0, m, 1);
	// nonlinear dependency vector
	VectorXi NL = VectorXi::Ones(n);
	for (int k = 0; k < n - m; k++)
		NL[k] = 0;
	// weights
	double alpha = 0.7;
	double lambda = alpha * alpha * m - m;
	double nl = m + lambda;
	auto sqrtnl = sqrt(nl);
	double W0 = lambda / nl;
	double W0cov = W0 + 1 + 2 - alpha * alpha;
	double Wi = 0.5 / nl;
	// partial Choleski
	MatrixXd L = PartialChol(S0, NL);
	L *= sqrtnl;
	// Map the sigma points
	std::vector<VectorXd> Zi;
	Zi.push_back(fin(xnl));
	for (int k = 0; k < m; k++)
		Zi.push_back(fin(xnl - L.block(n - m, k, m, 1)));
	for (int k = 0; k < m; k++)
		Zi.push_back(fin(xnl + L.block(n - m, k, m, 1)));
	int g = Zi[0].size();
	// Expected z value
	z = VectorXd::Zero(g);
	for (int k = 1; k < Zi.size(); k++)
		z += Zi[k];
	z *= Wi;
	z += W0 * Zi[0];
	// Sigma_z
	Sz = MatrixXd::Zero(g, g);
	{
		for (int k = 1; k < 2*m+1; k++) {
			auto temp = Zi[k] - z;
			Sz += temp * temp.transpose();
		}
		Sz *= Wi;
		auto temp = Zi[0] - z;
		Sz += W0cov * temp*temp.transpose();
	}
	// Sigma _zx
	Szx = MatrixXd::Zero(g, n);
	{
		for (int k = 0; k < m; k++)
			Szx += (Zi[k + 1 + m] - Zi[k + 1]) * L.block(0, k, 11, 1).transpose();
		Szx *= Wi;
	}
}

void SelUT(const VectorXd& x0, const MatrixXd& S0, int m, VectorXd(*fin)(const VectorXd&), const MatrixXd& A,
	VectorXd& y, MatrixXd& Sy, MatrixXd& Syx) {
	VectorXd z;
	MatrixXd Sz, Szx;
	UT(x0, S0, m, fin, z, Sz, Szx);
	int n = x0.size();
	int g = z.size();
	auto xl = x0.block(0, 0, n - m, 1);
	y = z + A * xl;
	Syx = Szx + A * S0.block(0, 0, n - m, n);
	Sy = Sz + Syx.block(0, 0, g, n - m) * A.transpose() + (Szx.block(0, 0, g, n - m) * A.transpose()).transpose();
}

int main (void) {
	// Deriving matrix A
	MatrixXd A = MatrixXd::Zero(11, 6);
	for (int i = 0; i < 6; i++)
		A(i, i) = 1;
	A(6, 2) = Ts;
	// Deriving matrix C
	MatrixXd C = MatrixXd::Zero(6, 6);
	A(0, 0) = 1;
	A(1, 1) = 1;
	A(3, 0) = 1;
	A(4, 1) = 1;
	A(2, 3) = 1;
	A(3, 4) = 1;
	A(4, 5) = 1;
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
	VectorXd xl = x.block(0, 0, 6, 1);
	VectorXd xnl = x.block(6, 0, 5, 1);

	//std::cout << f(x) - (A*xl + f2(xnl)) << std::endl;

	std::vector<VectorXd> saved;
	long N = 1e7;
	auto start = std::chrono::system_clock::now();
	for (int n = 0; n < N; n++)
		saved.push_back(f(x));
	auto end = std::chrono::system_clock::now();
	long dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("purely nonlinear function : %zd \n", dur / 1000);

	start = std::chrono::system_clock::now();
	for (int n = 0; n < N; n++)
		saved.push_back(A*xl + f2(xnl));
	end = std::chrono::system_clock::now();
	dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("mixed function = %zd \n", dur / 1000);

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
	
	long N_UT = 1e6;
	{
		// Original UT
		VectorXd origninal_y;
		MatrixXd origninal_Sy, origninal_Syx;
		start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++) {
			UT(x, S0, 11, f, origninal_y, origninal_Sy, origninal_Syx);
		}
		end = std::chrono::system_clock::now();
		dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("original UT : %zd \n", dur / 1000);

		// Modified UT
		VectorXd modified_y;
		MatrixXd modified_Syx, modified_Sy;
		start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++)
			SelUT(x, S0, 5, f2, A, modified_y, modified_Sy, modified_Syx);
		end = std::chrono::system_clock::now();
		dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("modified UT : %zd \n", dur/1000);

		//std::cout << (origninal_y - modified_y).norm() / origninal_y.norm() << std::endl;

		//std::cout << (origninal_Sy - modified_Sy).norm() / origninal_Sy.norm() << std::endl;

		//std::cout << (origninal_Syx - modified_Syx).norm() / origninal_Syx.norm() << std::endl;

		//std::cout << std::endl << origninal_y << std::endl << std::endl << modified_y << std::endl;

		//std::cout << std::endl << origninal_Sy << std::endl << std::endl << modified_Sy << std::endl;

		//std::cout << std::endl << origninal_Syx << std::endl << std::endl << modified_Syx << std::endl;
	}
	{
		// Original UKF pred
		VectorXd original_x1, original_y;
		MatrixXd origninal_Sx1, origninal_Sx1x0, origninal_Sy, origninal_Syx1;
		start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++) {
			UT(x, S0, 11, f, original_x1, origninal_Sx1, origninal_Sx1x0);
			UT(original_x1, origninal_Sx1, 11, g, original_y, origninal_Sy, origninal_Syx1);
		}
		end = std::chrono::system_clock::now();
		dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("original UT2 : %zd \n", dur / 1000);

		// Modified UKF pred
		VectorXd modified_x1, modified_y;
		MatrixXd modified_Sx1, modified_Sx1x0, modified_Syx, modified_Sy;
		start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++) {
			SelUT(x, S0, 5, f2, A, modified_x1, modified_Sx1, modified_Sx1x0);
			SelUT(modified_x1, modified_Sx1, 5, g2, C, modified_y, modified_Sy, modified_Syx);
		}
		end = std::chrono::system_clock::now();
		dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("modified UT2 : %zd \n", dur / 1000);
	}
	return 0;
}
