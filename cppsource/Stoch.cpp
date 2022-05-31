#include "Stoch.h"
#include <random>

Eigen::VectorXd UT::RandomVector(const Eigen::MatrixXd& S) {
	static std::default_random_engine generator;
	static std::normal_distribution<double> distribution(0, 1);
	Eigen::VectorXd out(S.cols());
	for (int i = 0; i < S.cols(); i++)
		out(i) = sqrt(S(i, i)) * distribution(generator);
	return out;
}

UT::ValWithCov::ValWithCov(const Eigen::VectorXd y_, const Eigen::MatrixXd& Sy_, const Eigen::MatrixXd& Sxy_) : y(y_), Sy(Sy_), Sxy(Sxy_) {}

UT::ValWithCov::ValWithCov(const Eigen::VectorXd y_, const Eigen::MatrixXd& Sy_) : y(y_), Sy(Sy_) {}
