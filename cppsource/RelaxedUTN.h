#pragma once

#include <vector>
#include "SelUT.h"

namespace RelaxedUnscentedTransformation {
	/*! \brief Relaxed Unscented transformation (without reindexing or exact subspace)
	*
	* Considering a function y= [I; F]*f( {x0,x1,x2,x3,...} ) + sum_n An*xn(il), where
	*  - f depends only on the values of vector xn with indices inl_n,
	* the algorithm approximates the expected value of y,
	* its covariance matrix Sy, and cross covariance matrix Sxny
	* from the expected value of xn and its covariance matrix Sxn
	*
	* Inputs:
	*   values : list as { x0,x1,..}, where xn is a vector of length nn
	*   ils: list as { x0,x1,... }, where iln contains indices of values of xn,
	*      the mapping depends on linearly (length: kn)
	*   inls: list as { x0,x1,... }, where  inln is the indices of values of xn,
	*      the nonlinear part depends on linearly (length: mn)
	*   An: list of matrices, such that An is a matrix of linear coefficients of size g x kn
	*   F: coefficients for f(x) with size (g-l) x l
	*   covariances: list of Sxn matrices (symmetric, poisitive definite) of size nn x nn
	*   fin: a function  z=fin( {x0,x1,x2,x3,...} ) {R^n0 x R^n1 x...} ->R^l (that can be a C callback, std::function, etc.)
	*
	* Outputs:
	*   z: a vector of size g
	*   Sz: a matrix of size g x g
	*  Sxz: list of matrices of size nn x g
	*/
	template <typename Func>
	void RelaxedUTN(const std::vector<Eigen::MatrixXd>& As, const std::vector<Eigen::VectorXi>& ils,
		Func fin, const Eigen::MatrixXd& F, const std::vector<Eigen::VectorXi>& inls,
		const std::vector<Eigen::VectorXd>& values, const std::vector<Eigen::MatrixXd>& covariances,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, std::vector<Eigen::MatrixXd>& Sxys);

	/*! \brief Selective Unscented transformation
	*
	* Considering a function z=f(x), where f depends only on values of x with given indices,
	* the algorithm approximates the expected value of z,
	* its covariance matrix Sz, and cross covariance matrix Sxz
	* from the expected value of x and its covariance matrix Sx
	*
	* Inputs:
	*   values : list as { x0,x1,..}, where xn is a vector of length nn
	*   inls: list as { x0,x1,... }, where  inln is the indices of values of xn,
	*      the nonlinear part depends on linearly (length: mn)
	*   covariances: list of Sxn matrices (symmetric, poisitive definite) of size nn x nn
	*   fin: a function  z=fin( {x0,x1,x2,x3,...} ) {R^n0 x R^n1 x...} ->R^l (that can be a C callback, std::function, etc.)
	*
	* Outputs:
	*   z: a vector of size g
	*   Sz: a matrix of size g x g
	*  Sxz: list of matrices of size nn x g
	*/
	template<typename Func>
	void SelUTN(const std::vector<Eigen::VectorXd>& values,
		const std::vector<Eigen::MatrixXd>& covariances, const std::vector<Eigen::VectorXi>& inls,
		Func fin, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, std::vector<Eigen::MatrixXd>& Sxz);

	template<typename Func>
	void SelUTN(const std::vector<Eigen::VectorXd>& values,
		const std::vector<Eigen::MatrixXd>& covariances, const std::vector<Eigen::VectorXi>& inls,
		Func fin, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, std::vector<Eigen::MatrixXd>& Sxz) {
		Sxz.clear();
		auto N = values.size();

		// Compute m0 as sum of lengthes of inl
		int m0 = 0;
		for (int n = 0; n < N; n++)
			m0 += (int)inls[n].size();
		// 
		for (int n = 0; n < N; n++) {
			// define fin_
			auto fin_ = [values, fin, n](const Eigen::VectorXd& nthValue)->Eigen::VectorXd {
				auto modValues = values;
				modValues[n] = nthValue;
				return fin(modValues);
			};
			Eigen::VectorXd z_;
			Eigen::MatrixXd Sz_, Sxz_;
			SelUT(values[n], covariances[n], inls[n], fin_, z_, Sz_, Sxz_, m0);
			if (n == 0) {
				z = z_;
				Sz = Sz_;
			}
			else {
				z += z_;
				Sz += Sz_;
			}
			Sxz.push_back(Sxz_);
		}
	}

	template <typename Func>
	void RelaxedUTN(const std::vector<Eigen::MatrixXd>& As, const std::vector<Eigen::VectorXi>& ils,
		Func fin, const Eigen::MatrixXd& F, const std::vector<Eigen::VectorXi>& inls,
		const std::vector<Eigen::VectorXd>& values, const std::vector<Eigen::MatrixXd>& covariances,
		Eigen::VectorXd& y, Eigen::MatrixXd& Sy, std::vector<Eigen::MatrixXd>& Sxys) {

		auto N = values.size();
		// UT based nonlinear part
		SelUTN(values, covariances, inls, fin, y, Sy, Sxys);

		// Determine b related quantities
		// b
		auto g = y.size();
		{
			Eigen::VectorXd b(g + F.rows());
			b.segment(0, g) = y;
			b.segment(g, F.rows()) = F * y;
			y = b;
		}
		auto g1 = y.size();
		// Sb
		{
			Eigen::MatrixXd Sb(g1, g1);
			Sb.block(0, 0, g, g) = Sy;
			auto temp = F * Sy;
			Sb.block(g, 0, F.rows(), g) = temp;
			Sb.block(0, g, g, F.rows()) = temp.transpose();
			Sb.block(g, g, F.rows(), F.rows()) = temp * F.transpose();
			Sy = Sb;
		}
		// Sbxn
		for (int n = 0; n < N; n++) {
			auto nn = values[n].size();
			Eigen::MatrixXd Sxnb(nn, g1);
			Sxnb.block(0, 0, nn, g) = Sxys[n];
			Sxnb.block(0, g, nn, F.rows()) = Sxys[n] * F.transpose();
			Sxys[n] = Sxnb;
		}

		// Add the effect of the linear parts
		for (int n = 0; n < N; n++) {
			// Init x_l
			Eigen::VectorXd xl = VectorSelect(values[n], ils[n]);

			// Add linear parts
			y += As[n] * xl;

			auto At = As[n].transpose();
			auto temp = Sxys[n];
			Sxys[n] += MatrixColumnSelect(covariances[n], ils[n]) * At;
			Sy += As[n] * MatrixRowSelect(Sxys[n], ils[n])
				+ MatrixColumnSelect(temp.transpose(), ils[n])*At;
		}
	}

}
