#pragma once

#include "Eigen/Dense"
#include <functional>
#include <vector>

class SLAMStateUpdate {
	Eigen::MatrixXd A, F;
	Eigen::VectorXi inl, il;

	void _updateN(int N);

public:
	SLAMStateUpdate();

	void UT(double Ts, double v, double omega, double Sv, double Somega,
		const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy);
};

struct SLAMOutputUpdate {
	Eigen::VectorXi il;

public:
	SLAMOutputUpdate();

	void UT(const std::vector<int>& actives, Eigen::VectorXd& x, const Eigen::MatrixXd& Sx,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy);

	void AUT(const std::vector<int>& actives, Eigen::VectorXd& x,
		const Eigen::MatrixXd& Sx, const Eigen::VectorXd& ymeas,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy);
};

Eigen::VectorXd SLAMStateUpdateFull(const Eigen::VectorXd& a, double Ts);

std::function<Eigen::VectorXd(const Eigen::VectorXd&)> SLAM_output_full_fcn(const std::vector<int>& actives);

