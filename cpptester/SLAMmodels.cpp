#include "SLAMmodels.h"
#include "RelaxedAUT.h"
#include "RelaxedUT.h"
using namespace Eigen;
using namespace RelaxedUnscentedTransformation;

VectorXd SLAMStateUpdateFull(const VectorXd& a, double Ts) {
	VectorXd out = a.segment(2, a.size() - 2);
	out(0) += a(0)*cos(a(4))*Ts;
	out(1) += a(0)*sin(a(4))*Ts;
	out(2) += a(1)*Ts;
	return out;
}

std::function<Eigen::VectorXd(const Eigen::VectorXd&)> SLAM_output_full_fcn(const std::vector<int>& actives) {

	auto slamFullModel = [actives](const Eigen::VectorXd& x)->Eigen::VectorXd {
		Eigen::VectorXd out(actives.size() * 2);
		for (int i = 0; i < actives.size(); i++) {
			double dx = x[3 + 2 * actives[i]] - x[0];
			double dy = x[3 + 2 * actives[i] + 1] - x[1];
			out(2 * i) = sqrt(dx*dx + dy * dy);
			out(2 * i + 1) = atan2(dy, dx) - x[2];
			while (out(2 * i + 1) < -EIGEN_PI)
				out(2 * i + 1) += 2 * EIGEN_PI;
			while (out(2 * i + 1) > EIGEN_PI)
				out(2 * i + 1) -= 2 * EIGEN_PI;
		}
		return out;
	};
	
	return slamFullModel;
}

void SLAMStateUpdate::_updateN(int N) {
	A = MatrixXd::Identity(2 * N + 3, 2 * N + 3);
	A(0, 0) = 0;
	A(1, 1) = 0;
	A(2, 2) = 0;
	A(0, 1) = 1;
	A(1, 2) = 1;
	// i_l = [1 2 3 5 6 ... 2N+4]
	il = VectorXi(2 * N + 3);
	il(0) = 1;
	il(1) = 2;
	il(2) = 3;
	for (int n = 0; n < 2 * N; n++)
		il(n + 3) = 5 + n;
	// K : 0 of 2N+2 x 3
	F = MatrixXd::Zero(2 * N, 3);
}

SLAMStateUpdate::SLAMStateUpdate() : inl(2) {
	// init A for 0 registered landmarks
	_updateN(0);
	// init i_nl = [0 4]
	inl(0) = 0;
	inl(1) = 4;
}

void SLAMStateUpdate::UT(double Ts, double v, double omega, double Sv, double Somega, const Eigen::VectorXd & x, const Eigen::MatrixXd & Sx, Eigen::VectorXd & y, Eigen::MatrixXd & Sy, Eigen::MatrixXd & Sxy) {
	int N = ((int)x.size() - 3) / 2;
	// Check size of A
	if (A.cols() != 2 * N + 3)
		_updateN(N);
	// Rewrite Ts into A
	A(2, 0) = Ts;

	auto fin = [Ts](const Eigen::VectorXd& x)->Eigen::VectorXd {
		VectorXd out(3);
		out(0) = x(0)*cos(x(4))*Ts;
		out(1) = x(0)*sin(x(4))*Ts;
		out(2) = x(4);
		return out;
	};
	Eigen::VectorXd in(x.size() + 2);
	Eigen::MatrixXd Sin = Eigen::MatrixXd::Zero(x.size() + 2, x.size() + 2);
	in(0) = v;
	in(1) = omega;
	in.segment(2, x.size()) = x;
	Sin(0, 0) = Sv;
	Sin(1, 1) = Somega;
	Sin.block(2, 2, x.size(), x.size()) = Sx;
	RelaxedUT(A, il, fin, F, inl, in, Sin, y, Sy, Sxy);
}

SLAMOutputUpdate::SLAMOutputUpdate() : il(1) {
	// i_l = [active(0)*2+[0 1] .... actives(end)*2+[0 1]]
	//VectorXi il(1);
	il[0] = 2;
}

void SLAMOutputUpdate::UT(const std::vector<int>& actives, Eigen::VectorXd & x, const Eigen::MatrixXd & Sx, Eigen::VectorXd & y, Eigen::MatrixXd & Sy, Eigen::MatrixXd & Sxy) {
	// A
	MatrixXd A = MatrixXd::Zero(2 * actives.size(), 1);
	for (int i = 0; i < actives.size(); i++)
		A(2 * i + 1, 0) = -1;
	// i_nl = []
	VectorXi inl(2 + 2 * (int)actives.size());
	inl(0) = 0;
	inl(1) = 1;
	for (int i = 0; i < actives.size(); i++) {
		inl(2 + 2 * i) = 3 + 2 * (int)actives[i];
		inl(2 + 2 * i + 1) = 3 + 2 * (int)actives[i] + 1;
	}

	auto fin = [actives](const Eigen::VectorXd& x)->Eigen::VectorXd {
		Eigen::VectorXd out(actives.size() * 2);
		for (int i = 0; i < actives.size(); i++) {
			double dx = x[3 + 2 * actives[i]] - x[0];
			double dy = x[3 + 2 * actives[i] + 1] - x[1];
			out(2 * i) = sqrt(dx*dx + dy * dy);
			double v = atan2(dy, dx);
			while (v - x[2] < -EIGEN_PI)
				v += 2 * EIGEN_PI;
			while (v - x[2] > EIGEN_PI)
				v -= 2 * EIGEN_PI;
			out(2 * i + 1) = v;
		}
		return out;
	};

	// K : 0 of 2N+2 x 3
	MatrixXd F = MatrixXd::Zero(0, 2 * actives.size());

	RelaxedUT(A, il, fin, F, inl, x, Sx, y, Sy, Sxy);
}

void SLAMOutputUpdate::AUT(const std::vector<int>& actives,
	Eigen::VectorXd & x, const Eigen::MatrixXd & Sx,
	const Eigen::VectorXd& ymeas, Eigen::VectorXd & y,
	Eigen::MatrixXd & Sy, Eigen::MatrixXd & Sxy) {
	// A
	MatrixXd A = MatrixXd::Zero(2 * (int)actives.size(), 1);
	for (int i = 0; i < actives.size(); i++)
		A(2 * i + 1, 0) = -1;
	// i_nl = []
	VectorXi inl(2 + 2 * (int)actives.size());
	inl(0) = 0;
	inl(1) = 1;
	for (int i = 0; i < actives.size(); i++) {
		inl(2 + 2 * i) = 3 + 2 * (int)actives[i];
		inl(2 + 2 * i + 1) = 3 + 2 * (int)actives[i] + 1;
	}

	auto fin = [actives](const Eigen::VectorXd& x)->Eigen::VectorXd {
		Eigen::VectorXd out(actives.size() * 2);
		for (int i = 0; i < actives.size(); i++) {
			double dx = x[3 + 2 * actives[i]] - x[0];
			double dy = x[3 + 2 * actives[i] + 1] - x[1];
			out(2 * i) = sqrt(dx*dx + dy * dy);
			double v = atan2(dy, dx);
			while (v - x[2] < -EIGEN_PI)
				v += 2 * EIGEN_PI;
			while (v - x[2] > EIGEN_PI)
				v -= 2 * EIGEN_PI;
			out(2 * i + 1) = v;
		}
		return out;
	};

	// K : 0 of 2N+2 x 3
	MatrixXd F = MatrixXd::Zero(0, 2 * actives.size());

	RelAUT(A, il, fin, F, inl, x, Sx, ymeas, 0, 50, 0.5, y, Sy, Sxy);
}
