#include "Simulator.h"

Eigen::VectorXd computeNewLandmarkPosition(const Eigen::VectorXd & rdelta_x) {
	Eigen::VectorXd out(2);
	out(0) = rdelta_x(2) + rdelta_x(0)*cos(rdelta_x(1) + rdelta_x(4));
	out(1) = rdelta_x(3) + rdelta_x(0)*sin(rdelta_x(1) + rdelta_x(4));
	return out;
}

void Simulator::_registerLandMark(int barcode, double R, double phi) {
	// Add initial pose
	Eigen::VectorXd xy_lm;
	Eigen::MatrixXd Sxylm, Sx_xylm;
	Eigen::VectorXd in(x.size() + 2);
	in(0) = R;
	in(1) = phi;
	in.segment(2, x.size()) = x;
	Eigen::MatrixXd Sin = Eigen::MatrixXd::Zero(x.size() + 2, x.size() + 2);
	Sin(0, 0) = settings.Sr;
	Sin(1, 1) = settings.Sphi;
	Sin.block(2, 2, x.size(), x.size()) = Sx;
	UT(in, Sin, computeNewLandmarkPosition, xy_lm, Sxylm, Sx_xylm);
	Eigen::VectorXd newx(x.size() + 2);
	newx.segment(0, x.size()) = x;
	newx.segment(x.size(), 2) = xy_lm;
	Eigen::MatrixXd newSx = Eigen::MatrixXd::Zero(x.size() + 2, x.size() + 2);
	newSx.block(0, 0, x.size(), x.size()) = Sx;
	newSx.block(0, x.size(), x.size(), 2) = Sx_xylm.block(2, 0, x.size(), 2);
	newSx.block(x.size(), 0, 2, x.size()) = Sx_xylm.block(2, 0, x.size(), 2).transpose();
	newSx.block(x.size(), x.size(), 2, 2) = Sxylm;
	x = newx;
	Sx = newSx;
	registeredLandmarks.push_back(barcode);
}

void Simulator::_Step(double dT_sec, RunSetting set) {
	Eigen::VectorXd y;
	Eigen::MatrixXd Sy, Syx0;
	if (settings.cType == SimSettings::NewUT) {
		stateUpdater.UT(dT_sec, v, omega, settings.Sv, settings.Somega, x, Sx, y, Sy, Syx0);
		x = y;
		Sx = Sy;
	}
	else {
		Eigen::VectorXd in(x.size() + 2);
		in(0) = v;
		in(1) = omega;
		in.segment(2, x.size()) = x;
		Eigen::MatrixXd Sin = Eigen::MatrixXd::Zero(x.size() + 2, x.size() + 2);
		Sin(0, 0) = settings.Sv;
		Sin(1, 1) = settings.Somega;
		Sin.block(2, 2, x.size(), x.size()) = Sx;
		UT(in, Sin, [dT_sec](const Eigen::VectorXd& in)->Eigen::VectorXd {return SLAMStateUpdateFull(in, dT_sec); }, y, Sy, Syx0);
		x = y;
		Sx = Sy;
	}
	if (set == LOGPATH)
		logfile << "\t; " << t << " " << x(0) << " " << x(1) << " " << x(2) << " ...\n";
}

const Eigen::VectorXd & Simulator::getx() const {
	return x;
}

const Eigen::MatrixXd & Simulator::getSx() const {
	return Sx;
}
