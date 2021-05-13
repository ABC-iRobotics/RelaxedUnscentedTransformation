#pragma once

#include "UT.h"
#include <Eigen/SparseCore>
#include "index_selector.h"

// g and Q
template <typename Func>
void RelUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
	Func fin, const Eigen::MatrixXd& F, const Eigen::VectorXi& g,
	const Eigen::SparseMatrix<double>& Q, const Eigen::SparseMatrix<double>& Q1,
	const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy);

// no g and Q
template <typename Func>
void RelUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
	Func fin, const Eigen::MatrixXd& F,
	const Eigen::SparseMatrix<double>& Q, const Eigen::SparseMatrix<double>& Q1,
	const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy);

// g and no Q
template <typename Func>
void RelUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
	Func fin, const Eigen::MatrixXd& F, const Eigen::VectorXi& g, const Eigen::VectorXi& inl,
	const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy);

// no g and no Q
template <typename Func>
void RelUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
	Func fin, const Eigen::MatrixXd& F, const Eigen::VectorXi& inl,
	const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy);

struct MixedNonlin {
	Eigen::VectorXi i;
	Eigen::VectorXd M;
	MixedNonlin(Eigen::VectorXi i, Eigen::VectorXd M) : i(i), M(M) {};
};
typedef std::vector<MixedNonlin> MixedNonlinearityList;

void genQ(int n, const Eigen::VectorXi& inl, const MixedNonlinearityList& mix,
	Eigen::SparseMatrix<double>& Q, Eigen::SparseMatrix<double>& Q1);


// g and Q
template <typename Func>
void RelUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
	Func fin, const Eigen::MatrixXd& F, const Eigen::VectorXi& g,
	const Eigen::SparseMatrix<double>& Q, const Eigen::SparseMatrix<double>& Q1,
	const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy) {
	// Perform UT around a
	Eigen::VectorXd b0;
	Eigen::MatrixXd Sb0, Sxb0;
	int m = (int)Q1.rows();
	SelUT(x0, Q*S0*Q1.transpose(), m, fin, b0, Sb0, Sxb0, Q);
	// Determine b related quantities
	Eigen::VectorXd b(b0.size() + F.rows());
	b.segment(0, b0.size()) = b0;
	b.segment(b0.size(), F.rows()) = F * b0;
	Eigen::MatrixXd Sb(b.size(), b.size());
	Sb.block(0, 0, b0.size(), b0.size()) = Sb0;
	{
		auto temp = F * Sb0;
		Sb.block(b0.size(), 0, F.rows(), b0.size()) = temp;
		Sb.block(0, b0.size(), b0.size(), F.rows()) = temp.transpose();
		Sb.block(b0.size(), b0.size(), F.rows(), F.rows()) = temp * F.transpose();
	}
	Eigen::MatrixXd Sxb(x0.size(), b.size());
	Sxb.block(0, 0, x0.size(), b0.size()) = Sxb0;
	Sxb.block(0, b0.size(), x0.size(), F.rows()) = Sxb0 * F.transpose();
	// Init x_l
	Eigen::VectorXd xl = VectorSelect(x0, il);
	// Determine y related quantities
	y = VectorSelect(b, g) + A * xl;
	{
		auto Sxbg = MatrixColumnSelect(Sxb, g);
		auto Sbgbg = MatrixColumnSelect(MatrixRowSelect(Sb, g), g);
		auto At = A.transpose();
		Sxy = Sxbg + MatrixColumnSelect(S0, il) * At;
		Sy = Sbgbg + A * MatrixRowSelect(Sxy, il) + MatrixColumnSelect(Sxbg.transpose(), il)*At;
	}
}

// no g and Q
template <typename Func>
void RelUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
	Func fin, const Eigen::MatrixXd& F,
	const Eigen::SparseMatrix<double>& Q, const Eigen::SparseMatrix<double>& Q1,
	const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy) {
	// Perform UT around a
	Eigen::VectorXd b0;
	Eigen::MatrixXd Sb0, Sxb0;
	int m = (int)Q1.rows();
	SelUT(x0, Q*S0*Q1.transpose(), m, fin, b0, Sb0, Sxb0, Q);
	// Determine b related quantities
	Eigen::VectorXd b(b0.size() + F.rows());
	b.segment(0, b0.size()) = b0;
	b.segment(b0.size(), F.rows()) = F * b0;
	Eigen::MatrixXd Sb(b.size(), b.size());
	Sb.block(0, 0, b0.size(), b0.size()) = Sb0;
	{
		auto temp = F * Sb0;
		Sb.block(b0.size(), 0, F.rows(), b0.size()) = temp;
		Sb.block(0, b0.size(), b0.size(), F.rows()) = temp.transpose();
		Sb.block(b0.size(), b0.size(), F.rows(), F.rows()) = temp * F.transpose();
	}
	Eigen::MatrixXd Sxb(x0.size(), b.size());
	Sxb.block(0, 0, x0.size(), b0.size()) = Sxb0;
	Sxb.block(0, b0.size(), x0.size(), F.rows()) = Sxb0 * F.transpose();
	// Init x_l
	Eigen::VectorXd xl = VectorSelect(x0, il);
	// Determine y related quantities
	y = b + A * xl;
	{
		auto At = A.transpose();
		Sxy = Sxb + MatrixColumnSelect(S0, il) * At;
		Sy = Sb + A * MatrixRowSelect(Sxy, il) + MatrixColumnSelect(Sxb.transpose(), il)*At;
	}
}

// g and no Q
template <typename Func>
void RelUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
	Func fin, const Eigen::MatrixXd& F, const Eigen::VectorXi& g, const Eigen::VectorXi& inl,
	const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy) {
	// Perform UT around a
	Eigen::VectorXd b0;
	Eigen::MatrixXd Sb0, Sxb0;
	int m = (int)Q1.rows();
	SelUT(x0, S0, inl, fin, b0, Sb0, Sxb0);
	// Determine b related quantities
	Eigen::VectorXd b(b0.size() + F.rows());
	b.segment(0, b0.size()) = b0;
	b.segment(b0.size(), F.rows()) = F * b0;
	Eigen::MatrixXd Sb(b.size(), b.size());
	Sb.block(0, 0, b0.size(), b0.size()) = Sb0;
	{
		auto temp = F * Sb0;
		Sb.block(b0.size(), 0, F.rows(), b0.size()) = temp;
		Sb.block(0, b0.size(), b0.size(), F.rows()) = temp.transpose();
		Sb.block(b0.size(), b0.size(), F.rows(), F.rows()) = temp * F.transpose();
	}
	Eigen::MatrixXd Sxb(x0.size(), b.size());
	Sxb.block(0, 0, x0.size(), b0.size()) = Sxb0;
	Sxb.block(0, b0.size(), x0.size(), F.rows()) = Sxb0 * F.transpose();
	// Init x_l
	Eigen::VectorXd xl = VectorSelect(x0, il);
	// Determine y related quantities
	y = VectorSelect(b, g) + A * xl;
	{
		auto Sxbg = MatrixColumnSelect(Sxb, g);
		auto Sbgbg = MatrixColumnSelect(MatrixRowSelect(Sb, g), g);
		auto At = A.transpose();
		Sxy = Sxbg + MatrixColumnSelect(S0, il) * At;
		Sy = Sbgbg + A * MatrixRowSelect(Sxy, il) + MatrixColumnSelect(Sxbg.transpose(), il)*At;
	}
}

// no g and no Q
template <typename Func>
void RelUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
	Func fin, const Eigen::MatrixXd& F, const Eigen::VectorXi& inl,
	const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy) {
	// Perform UT around a
	Eigen::VectorXd b0;
	Eigen::MatrixXd Sb0, Sxb0;
	int m = (int)inl.size();
	SelUT(x0, S0, inl, fin, b0, Sb0, Sxb0);
	// Determine b related quantities
	Eigen::VectorXd b(b0.size() + F.rows());
	b.segment(0, b0.size()) = b0;
	b.segment(b0.size(), F.rows()) = F * b0;
	Eigen::MatrixXd Sb(b.size(), b.size());
	Sb.block(0, 0, b0.size(), b0.size()) = Sb0;
	{
		auto temp = F * Sb0;
		Sb.block(b0.size(), 0, F.rows(), b0.size()) = temp;
		Sb.block(0, b0.size(), b0.size(), F.rows()) = temp.transpose();
		Sb.block(b0.size(), b0.size(), F.rows(), F.rows()) = temp * F.transpose();
	}
	Eigen::MatrixXd Sxb(x0.size(), b.size());
	Sxb.block(0, 0, x0.size(), b0.size()) = Sxb0;
	Sxb.block(0, b0.size(), x0.size(), F.rows()) = Sxb0 * F.transpose();
	// Init x_l
	Eigen::VectorXd xl = VectorSelect(x0, il);
	// Determine y related quantities
	y = b + A * xl;
	{
		auto At = A.transpose();
		Sxy = Sxb + MatrixColumnSelect(S0, il) * At;
		Sy = Sb + A * MatrixRowSelect(Sxy, il) + MatrixColumnSelect(Sxb.transpose(), il)*At;
	}
}
