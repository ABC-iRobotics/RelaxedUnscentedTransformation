#pragma once
#include <Eigen/SparseCore>
#include "SelUT.h"

namespace RelaxedUnscentedTransformation {

	/*! \brief Relaxed Unscented transformation (with reindexing and exact subspace)
	*
	* Considering a function y=A*x(il) + z(g), where 
	*  - z(g) means the reindexing of vector z=[I; F]*f(x)
	*  - and f depends only on the first m values of vector (Q*x),
	* the algorithm approximates the expected value of y,
	* its covariance matrix Sy, and cross covariance matrix Sxy
	* from the expected value of x and its covariance matrix Sx
	*
	* Inputs:
	*   x: a vector of size n
	*   Q: an orthogonal matrix of size n x n (default value: identity matrix)
	*	Q1: the first m rows of matrix Q (size: m x n)
	*   g : index vector for reindexing (length: l)
	*   il: indices of values of x, the mapping depends on linearly (length: k)
	*   A: matrix of linear coefficients of size g x k
	*   F: coefficients for f(x) with size (g-l) x l
	*   Sx: a matrix (symmetric, poisitive definite) of size n x n
	*   fin: a function  z=fin(x) R^n->R^l (that can be a C callback, std::function, etc.)
	*   N: order of approximation (2,..,4 is allowed)
	*
	* Outputs:
	*   z: a vector of size g
	*   Sz: a matrix of size g x g
	*  Sxz: a matrix of size n x g
	*/
	template <typename Func>
	void RelaxedUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
		Func fin, int N, const Eigen::MatrixXd& F, const Eigen::VectorXi& g,
		const Eigen::SparseMatrix<double>& Q, const Eigen::SparseMatrix<double>& Q1,
		const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy);

	/*! \brief Relaxed Unscented transformation (with exact subspace)
	*
	* Considering a function y=A*x(il) + [I;F]*f(x), where
	*  - and f depends only on the first m values of vector (Q*x),
	* the algorithm approximates the expected value of y,
	* its covariance matrix Sy, and cross covariance matrix Sxy
	* from the expected value of x and its covariance matrix Sx
	*
	* Inputs:
	*   x: a vector of size n
	*   Q: an orthogonal matrix of size n x n (default value: identity matrix)
	*	Q1: the first m rows of matrix Q (size: m x n)
	*   il: indices of values of x, the mapping depends on linearly (length: k)
	*   A: matrix of linear coefficients of size g x k
	*   F: coefficients for f(x) with size (g-l) x l
	*   Sx: a matrix (symmetric, poisitive definite) of size n x n
	*   fin: a function  z=fin(x) R^n->R^l (that can be a C callback, std::function, etc.)
	*   N: order of approximation (2,..,4 is allowed)
	*
	* Outputs:
	*   z: a vector of size g
	*   Sz: a matrix of size g x g
	*  Sxz: a matrix of size n x g
	*/
	template <typename Func>
	void RelaxedUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
		Func fin, int N, const Eigen::MatrixXd& F,
		const Eigen::SparseMatrix<double>& Q, const Eigen::SparseMatrix<double>& Q1,
		const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy);

	/*! \brief Relaxed Unscented transformation (with reindexing)
	*
	* Considering a function y=A*x(il) + z(g), where
	*  - z(g) means the reindexing of vector z=[I; F]*f(x)
	*  - and f depends only on the values of vector x with indices inl,
	* the algorithm approximates the expected value of y,
	* its covariance matrix Sy, and cross covariance matrix Sxy
	* from the expected value of x and its covariance matrix Sx
	*
	* Inputs:
	*   x: a vector of size n
	*   g : index vector for reindexing (length: l)
	*   il: indices of values of x, the mapping depends on linearly (length: k)
	*   inl: indices of values of x, the nonlinear part depends on linearly (length: m)
	*   A: matrix of linear coefficients of size g x k
	*   F: coefficients for f(x) with size (g-l) x l
	*   Sx: a matrix (symmetric, poisitive definite) of size n x n
	*   fin: a function  z=fin(x) R^n->R^l (that can be a C callback, std::function, etc.)
	*   N: order of approximation (2,..,4 is allowed)
	*
	* Outputs:
	*   z: a vector of size g
	*   Sz: a matrix of size g x g
	*  Sxz: a matrix of size n x g
	*/
	template <typename Func>
	void RelaxedUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
		Func fin, int N, const Eigen::MatrixXd& F, const Eigen::VectorXi& g, const Eigen::VectorXi& inl,
		const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy);

	/*! \brief Relaxed Unscented transformation (without reindexing or exact subspace)
	*
	* Considering a function y=A*x(il) + [I; F]*f(x), where
	*  - f depends only on the values of vector x with indices inl,
	* the algorithm approximates the expected value of y,
	* its covariance matrix Sy, and cross covariance matrix Sxy
	* from the expected value of x and its covariance matrix Sx
	*
	* Inputs:
	*   x: a vector of size n
	*   il: indices of values of x, the mapping depends on linearly (length: k)
	*   inl: indices of values of x, the nonlinear part depends on linearly (length: m)
	*   A: matrix of linear coefficients of size g x k
	*   F: coefficients for f(x) with size (g-l) x l
	*   Sx: a matrix (symmetric, poisitive definite) of size n x n
	*   fin: a function  z=fin(x) R^n->R^l (that can be a C callback, std::function, etc.)
	*   N: order of approximation (2,..,4 is allowed)
	*
	* Outputs:
	*   z: a vector of size g
	*   Sz: a matrix of size g x g
	*  Sxz: a matrix of size n x g
	*/
	template <typename Func>
	void RelaxedUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
		Func fin, int N, const Eigen::MatrixXd& F, const Eigen::VectorXi& inl,
		const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy);

	/*! \brief Type to describe if the function depends on the M*x(i) combination of the values of x with indices i */
	struct MixedNonlin {
		Eigen::VectorXi i;
		Eigen::VectorXd M;
		MixedNonlin(Eigen::VectorXi i, Eigen::VectorXd M) : i(i), M(M) {};
	};
	/*! \brief Type to describe a list of such a dependencies*/
	typedef std::vector<MixedNonlin> MixedNonlinearityList;

	/*! \brief Algorithm to determine Q and Q1 matrices for a given dependency list*/
	void genQ(int n, const Eigen::VectorXi& inl, const MixedNonlinearityList& mix,
		Eigen::SparseMatrix<double>& Q, Eigen::SparseMatrix<double>& Q1);


	// g and Q
	template <typename Func>
	void RelaxedUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
		Func fin, int N, const Eigen::MatrixXd& F, const Eigen::VectorXi& g,
		const Eigen::SparseMatrix<double>& Q, const Eigen::SparseMatrix<double>& Q1,
		const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy) {
		// Perform UT around a
		Eigen::VectorXd b0;
		Eigen::MatrixXd Sb0, Sxb0;
		int m = (int)Q1.rows();
		SelUT(x0, Q*S0*Q1.transpose(), m, fin, N, b0, Sb0, Sxb0, Q);
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
	void RelaxedUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
		Func fin, int N, const Eigen::MatrixXd& F,
		const Eigen::SparseMatrix<double>& Q, const Eigen::SparseMatrix<double>& Q1,
		const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy) {
		// Perform UT around a
		Eigen::VectorXd b0;
		Eigen::MatrixXd Sb0, Sxb0;
		int m = (int)Q1.rows();
		SelUT(x0, Q*S0*Q1.transpose(), m, fin, N, b0, Sb0, Sxb0, Q);
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
	void RelaxedUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
		Func fin, int N, const Eigen::MatrixXd& F, const Eigen::VectorXi& g, const Eigen::VectorXi& inl,
		const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy) {
		// Perform UT around a
		Eigen::VectorXd b0;
		Eigen::MatrixXd Sb0, Sxb0;
		int m = (int)Q1.rows();
		SelUT(x0, S0, inl, fin, N, b0, Sb0, Sxb0);
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
	void RelaxedUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
		Func fin, int N, const Eigen::MatrixXd& F, const Eigen::VectorXi& inl,
		const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy) {
		// Perform UT around a
		Eigen::VectorXd b0;
		Eigen::MatrixXd Sb0, Sxb0;
		int m = (int)inl.size();
		SelUT(x0, S0, inl, fin, N, b0, Sb0, Sxb0);
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

}