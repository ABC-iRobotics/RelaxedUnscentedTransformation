#include "Stoch.h"
#include <random>

Eigen::VectorXd UT::RandomVector(const Eigen::MatrixXd& sqrtS) {
	static std::default_random_engine generator;
	static std::normal_distribution<double> distribution(0, 1);
	Eigen::VectorXd out(sqrtS.cols());
	for (int i = 0; i < sqrtS.cols(); i++)
		out(i) = distribution(generator);
	return sqrtS * out;
}

UT::ValWithCov::ValWithCov() : y(0), Sy(0, 0), Sxy(0, 0) {};

UT::ValWithCov::ValWithCov(const Eigen::VectorXd y_, const Eigen::MatrixXd& Sy_, const Eigen::MatrixXd& Sxy_) : y(y_), Sy(Sy_), Sxy(Sxy_) {}

UT::ValWithCov::ValWithCov(const Eigen::VectorXd y_, const Eigen::MatrixXd& Sy_) : y(y_), Sy(Sy_) {}
