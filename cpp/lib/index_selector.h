#pragma once

#include "Eigen/Dense"

Eigen::VectorXd VectorSelect(const Eigen::VectorXd& v,
	const Eigen::VectorXi& indices);

Eigen::MatrixXd MatrixRowSelect(const Eigen::MatrixXd& m,
	const Eigen::VectorXi& indices);

Eigen::MatrixXd MatrixColumnSelect(const Eigen::MatrixXd& m,
	const Eigen::VectorXi& indices);

Eigen::MatrixXd ProductAlong(const Eigen::MatrixXd& m0,
	const Eigen::VectorXi& ic0, const Eigen::MatrixXd& m1);