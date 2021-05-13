#include "PartialCholevski.h"
#include<sstream>
using namespace RelaxedUT;

typedef Eigen::LLT<Eigen::MatrixXd>::RealScalar Real;

Eigen::MatrixXd RelaxedUT::PartialChol(Eigen::MatrixXd a, Eigen::VectorXi v) {
	const Eigen::Index n = a.rows();
	double eps = a.trace()*1e-10;
	const Eigen::Index nOut = v.sum();
	Eigen::MatrixXd out = Eigen::MatrixXd(n, nOut);
	unsigned int i = 0;
	unsigned int nSkipped = 0;
	for (unsigned int j = 0; j < n; j++)
		if (v[j] == 1) {
			Real temp = a(j, j);
			for (unsigned int k = 0; k < i; k++)
				temp -= out(j, k) * out(j, k);
			if (temp < eps && temp>-eps) {
				nSkipped++;
				break;
			}
			if (temp < Real(0)) {

				std::stringstream errormsg;
				errormsg << "The value " << temp << " is negative, when matrix (" << a <<
					") was  decomposed by column (" << v.transpose() <<
					"). The matrix is not positive definite or numerical error.";
				throw std::runtime_error(errormsg.str());
			}
			out(j, i) = sqrtl(temp);
			for (unsigned int k = 0; k < n; k++) {
				if (v[k] == 1 && k < j)
					out(k, i) = 0;
				else if (k != j) {
					temp = a(k, j);
					for (unsigned int l = 0; l < i; l++)
						temp -= out(k, l)*out(j, l);
					temp /= out(j, i);
					out(k, i) = temp;
				}
			}
			i++;
		}
	return out.block(0,0,n,nOut-nSkipped);
}
