#include "SLAMexample.h"
#include "RelUT.h"
#include "UT.h"
#include <iostream>
#include "Eigen/Dense"
#include <chrono>

using namespace RelaxedUT;

typedef std::chrono::time_point<std::chrono::system_clock> Time;

typedef std::chrono::microseconds DTime;

DTime duration_since_epoch(const Time& t);

template<class _Rep, class _Period>
inline DTime duration_cast(const std::chrono::duration<_Rep, _Period>& in);

template<class _Rep, class _Period>
inline double duration_cast_to_sec(const std::chrono::duration<_Rep, _Period>& in);

Time InitFromDurationSinceEpochInMicroSec(const long long& value);

DTime InitFromDurationInMicroSec(const long long& value);

Time Now();

template<class _Rep, class _Period>
inline DTime duration_cast(const std::chrono::duration<_Rep, _Period>& in) {
	return std::chrono::duration_cast<DTime>(in);
}

template<class _Rep, class _Period>
inline double duration_cast_to_sec(const std::chrono::duration<_Rep, _Period>& in) {
	return duration_cast(in).count() / 1e6;
}

using namespace Eigen;

struct RelaxedIdentityUT {
	VectorXi il, g;
	MatrixXd A, F;
	Eigen::SparseMatrix<double> Q, Q1;

	// x -> b0
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
		RelUT(A, il, RelaxedIdentityUT::IdentityExampleNonlinear, F, g, Q, Q1,
			x, Sx, y, Sy, Sxy);
	}
};


VectorXd IdentityFull(const VectorXd& a) {
	return a;
}

int main() {
	auto mixed_mapping = RelaxedIdentityUT();
	Eigen::VectorXd x(4);
	x << 0.1, 0.2, 0.3, 0.4;

	MatrixXd Sx = MatrixXd::Identity(4, 4);
	Sx(2, 2) = 0.2;

	Eigen::VectorXd y1;
	Eigen::MatrixXd Sy1, Sxy1;
	long N = 1000000;
	{
		auto start = std::chrono::system_clock::now();
		for (long i = 0; i < N; i++)
			mixed_mapping.UT(x, Sx, y1, Sy1, Sxy1);
		auto end = std::chrono::system_clock::now();
		long dur = (long)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("new UT : %ld \n", dur / 1000);
	}
	Eigen::VectorXd y2;
	Eigen::MatrixXd Sy2, Sxy2;
	{
		auto start = std::chrono::system_clock::now();
		for (long i = 0; i < N; i++)
			UT(x, Sx, IdentityFull, y2, Sy2, Sxy2);
		auto end = std::chrono::system_clock::now();
		long dur = (long)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("original UT : %ld \n", dur / 1000);
	}
	std::cout << y1 << std::endl << std::endl;
	std::cout << y2 << std::endl << std::endl;
	//std::cout << y1 - y2 << std::endl;
	std::cout << Sy1 << std::endl << std::endl;
	std::cout << Sy2 << std::endl << std::endl;
	//std::cout << Sy1 - Sy2 << std::endl;

	std::cout << Sxy1 << std::endl << std::endl;
	std::cout << Sxy2 << std::endl << std::endl;
	//std::cout << Sxy1 - Sxy2 << std::endl << std::endl;

	return 0;
}