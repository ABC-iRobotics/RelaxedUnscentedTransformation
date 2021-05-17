#include "RelaxedUT.h"
#include "UT.h"
#include <iostream>

using namespace RelaxedUnscentedTransformation;
using namespace Eigen;

// This example considers the most simple, y=x mapping where the length of vector x is 4

// The original UT is used on the following implementation of the function:
VectorXd IdentityFull(const VectorXd& a) {
	return a;
}

// In order to show the precision of the relaxed method (by using it in a so complicated way as possible)
// The following form of the mapping is used:
//
// y = A*x_lin + z(g)
// where
//  A = [0 0   0 0;
//       0 0  -1 0;
//      -1 -2 -1 0;
//       0 0   0 1];
//  x_lin = [x(i_lin(0)); x(i_lin(1)); x(i_lin(2)); x(i_lin(3));]; i_lin = [0 1 2 3];
//  z(g) means [z(g(0)); z(g(1)); z(g(2)); z(g(3))]; g = [0 1 2 3];
//  and the original z vector is
//  z = [ eye(3,3) ] * f_nonlin
//      [ F        ]
//  f_nonlin = [x(0); x(1)+x(2); x(0)+2x(1)+2x(2)];
//  this nonlinearity is described as i_nl = [0] and
//  two groups as i1 = [1 2] with m1=[ 1 1]
//          and i2 = [0 1 2] with m2=[1 2 2]
//
// The following struct initializes and stores this values for usage
struct RelaxedIdentityUT {
	VectorXi il, g;
	MatrixXd A, F;
	Eigen::SparseMatrix<double> Q, Q1;

	// z = [a(0); a(1)+a(2); a(0)+2a(1)+2a(2)]
	static VectorXd IdentityExampleNonlinear(const VectorXd& a) {
		VectorXd out(3);
		out(0) = a(0);
		out(1) = a(1) + a(2);
		out(2) = a(0) + 2 * a(1) + 2 * a(2);
		return out;
	}

	// Constructor
	RelaxedIdentityUT() : il(4), g(4) {
		il(0) = 0;
		il(1) = 1;
		il(2) = 2;
		il(3) = 3;
		// A
		A = MatrixXd::Zero(4, 4);
		A(1, 2) = -1;
		A(2, 0) = -1;
		A(2, 1) = -2;
		A(2, 2) = -1;
		A(3, 3) = 1;
		// g:1...4
		for (int n = 0; n < 4; n++)
			g(n) = n;
		// F : 0 of 2N+2 x 3
		F = MatrixXd::Zero(1, 3);
		{
			// i_nl = [0]
			VectorXi inl(1);
			inl(0) = 0;
			// m1
			VectorXd m1(2);
			m1(0) = 1; m1(1) = 1;
			VectorXi i1(2);
			i1(0) = 1; i1(1) = 2;
			// m2
			VectorXd m2(3);
			m2(0) = 1; m2(1) = 2; m2(2) = 2;
			VectorXi i2(3);
			i2(0) = 0; i2(1) = 1; i2(2) = 2;

			MixedNonlinearityList list = { MixedNonlin(i1, m1), MixedNonlin(i2, m2) };

			genQ(4, inl, list, Q, Q1);
		}
	}

	void UT(const VectorXd& x, const MatrixXd& Sx, VectorXd& y, MatrixXd& Sy, MatrixXd& Sxy) {
		RelaxedUT(A, il, RelaxedIdentityUT::IdentityExampleNonlinear, F, g, Q, Q1,
			x, Sx, y, Sy, Sxy);
	}
};


int main() {
	auto mixed_mapping = RelaxedIdentityUT();
	Eigen::VectorXd x(4);
	x << 0.1, 0.2, 0.3, 0.4;

	MatrixXd Sx = MatrixXd::Identity(4, 4);
	Sx(2, 2) = 0.2;

	// original UT method
	Eigen::VectorXd y2;
	Eigen::MatrixXd Sy2, Sxy2;
	UT(x, Sx, IdentityFull, y2, Sy2, Sxy2);

	// new UT method
	Eigen::VectorXd y1;
	Eigen::MatrixXd Sy1, Sxy1;
	mixed_mapping.UT(x, Sx, y1, Sy1, Sxy1);
	
	std::cout << "Conidering x=[ " << x.transpose() << "], and Sx =\n[" << Sx << "]\nand identity function y=f(x)\n\n";

	std::cout << "In the new method, the identity is implemented in the most complicated way, using reindexing and " <<
		"exact subspace etc., in order to show these functions in a very simple problem.\n\n";

	std::cout << "Expected value of f(x) with the new method and the original UT:\n";
	std::cout << "[" << y1.transpose() << "]" << std::endl;
	std::cout << "[" << y2.transpose() << "]" << std::endl << std::endl;
	
	std::cout << "Matrix Syy with the new method and the original UT:\n" << std::endl;
	std::cout << Sy1 << std::endl << std::endl;
	std::cout << Sy2 << std::endl << std::endl;

	std::cout << "Matrix Sxy with the new method and the original UT:\n" << std::endl;
	std::cout << Sxy1 << std::endl << std::endl;
	std::cout << Sxy2 << std::endl << std::endl;

	std::cout << std::endl << "All of them return the same (trivial) result with numeric errors\n";
	
	return 0;
}