#include "IndexSelector.h"
using namespace Eigen;

Eigen::VectorXd UT::VectorSelect(const Eigen::VectorXd& v,
	const Eigen::VectorXi& indices) {
	Eigen::VectorXd out(indices.size());
	for (int n = 0; n < indices.size(); n++)
		out(n) = v(indices(n));
	return out;
}

Eigen::MatrixXd UT::MatrixRowSelect(const Eigen::MatrixXd& m,
	const Eigen::VectorXi& indices) {
	Eigen::MatrixXd out(indices.size(), m.cols());
	for (int n = 0; n < indices.size(); n++)
		out.row(n) = m.row(indices(n));
	return out;
}

Eigen::MatrixXd UT::MatrixColumnSelect(const Eigen::MatrixXd& m,
	const Eigen::VectorXi& indices) {
	Eigen::MatrixXd out(m.rows(), indices.size());
	for (int n = 0; n < indices.size(); n++)
		out.col(n) = m.col(indices(n));
	return out;
}

Eigen::MatrixXd UT::ProductAlong(const Eigen::MatrixXd& m0, const VectorXi& ic0,
	const MatrixXd& m1) {
	auto R = m0.rows();
	auto C = m1.cols();
	Eigen::MatrixXd out(R, C);
	for (int r = 0; r < R; r++) {
		VectorXd row = m0.row(r);
		for (int c = 0; c < C; c++) {
			out(r, c) = 0;
			for (int k = 0; k < m1.cols(); k++)
				out(r, c) += row(ic0(k)) * (m1.col(c))(k);
		}
	}
	return out;
}