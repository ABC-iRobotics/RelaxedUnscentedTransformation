#pragma once

#include "Eigen/Dense"

namespace RelaxedUnscentedTransformation {

	/* \brief Function that returns the values of a vector with given indices as a new vector
	*/
	Eigen::VectorXd VectorSelect(const Eigen::VectorXd& v,
		const Eigen::VectorXi& indices);

	/* \brief Function that returns the rows of a matrix with given indices as a new matrix
	*/
	Eigen::MatrixXd MatrixRowSelect(const Eigen::MatrixXd& m,
		const Eigen::VectorXi& indices);

	/* \brief Function that returns the columns of a matrix with given indices as a new matrix
	*/
	Eigen::MatrixXd MatrixColumnSelect(const Eigen::MatrixXd& m,
		const Eigen::VectorXi& indices);

	/* \brief Function that returns product of MatrixColumnSelect(m0,ic0)*m1
	*/
	Eigen::MatrixXd ProductAlong(const Eigen::MatrixXd& m0,
		const Eigen::VectorXi& ic0, const Eigen::MatrixXd& m1);

}