#pragma once

#include "RelUT.h"
#include <Eigen/QR>

using namespace RelaxedUT;

void RelaxedUT::genQ(int n, const Eigen::VectorXi& inl, const MixedNonlinearityList& mix,
	Eigen::SparseMatrix<double>& Q,
	Eigen::SparseMatrix<double>& Q1) {
	// Construct M matrix from the weights keeping it orthogonal to inl
	typedef Eigen::MatrixXd Matrix;
	Matrix M = Matrix::Zero(mix.size() + inl.size(), n);
	for (int n = 0; n < inl.size(); n++)
		M(n, inl(n)) = 1;
	for (int n = 0; n < mix.size(); n++)
		for (int i0 = 0; i0 < mix[n].i.size(); i0++) {
			bool flag = false;
			for (int i = 0; i < inl.size(); i++)
				if (mix[n].i[i0] == inl(i))
					flag = true;
			if (!flag)
				M(n + inl.size(), mix[n].i[i0]) = mix[n].M(i0);
		}
	// RQ factorization
	Eigen::HouseholderQR<Matrix> solver(M.transpose());
	Eigen::MatrixXd Qdense = solver.householderQ().transpose();
	double rel_treshold = 1e-6;
	// Compute rank
	int rank = n;
	{
		Eigen::MatrixXd to_compute_rank = (M * Qdense.transpose()).cwiseAbs();
		double treshold = to_compute_rank.maxCoeff() * rel_treshold;
		for (int i = 0; i < to_compute_rank.cols(); i++)
			if (to_compute_rank.col(i).maxCoeff() < treshold) {
				rank = i;
				break;
			}
	}
	// make it sparse
	for (int i = 0; i < Qdense.rows(); i++)
		for (int j = 0; j < Qdense.cols(); j++)
			if (abs(Qdense(i, j)) < 1e-8)
				Qdense(i, j) = 0;
	// save it sparse
	Q = Qdense.sparseView();
	Q1 = Q.block(0, 0, rank, n);
}
