#include "index_selector.h"
#include <iostream>

using namespace Eigen;

Eigen::VectorXd VectorSelect(const Eigen::VectorXd& v,
	const Eigen::VectorXi& indices) {
	Eigen::VectorXd out(indices.size());
	for (int n = 0; n < indices.size(); n++)
		out(n) = v(indices(n));
	return out;
}

Eigen::MatrixXd MatrixRowSelect(const Eigen::MatrixXd& m,
	const Eigen::VectorXi& indices) {
	Eigen::MatrixXd out(indices.size(), m.cols());
	for (int n = 0; n < indices.size(); n++)
		out.row(n) = m.row(indices(n));
	return out;
}

Eigen::MatrixXd MatrixColumnSelect(const Eigen::MatrixXd& m,
	const Eigen::VectorXi& indices) {
	Eigen::MatrixXd out(m.rows(), indices.size());
	for (int n = 0; n < indices.size(); n++)
		out.col(n) = m.col(indices(n));
	return out;
}

Eigen::MatrixXd ProductAlong(const Eigen::MatrixXd& m0, const VectorXi& ic0,
	const MatrixXd& m1) {
	auto R = m0.rows();
	auto C = m1.cols();
	//if (m0.IsRowMajor == 0)
	//	std::cout << "row" << std::endl;
	//VectorXd a = m0.col(0);
	//VectorXd b = m0.row(0);
	Eigen::MatrixXd out(R, C);
	//out(0, 0) = a.cwiseProduct(a).sum() + b.cwiseProduct(b).prod();
	for (int r = 0; r < R; r++) {
		VectorXd row = m0.row(r);
		for (int c = 0; c < C; c++) {
			out(r, c) = 0;
			for (int k = 0; k < m1.cols(); k++) {
				//std::cout << row << std::endl << std::endl;
				//std::cout << ic0(k) << std::endl << std::endl;
				//std::cout << k << std::endl << std::endl;
				out(r, c) += row(ic0(k)) * (m1.col(c))(k);
			}
		}
	}
	/*
	for (int c = 0; c < C; c++) {
		out(r, c) = row.cwiseProduct(m1.col(c)).sum();



	}*/
	return out;
}