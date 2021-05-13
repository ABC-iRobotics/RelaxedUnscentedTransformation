#pragma once
#include <vector>
#include <map>

typedef std::map<int, int> Barcodes;

Barcodes readBarcodes();

struct Measurment {
	double t;
	int barcode;
	double R, phi;
	Measurment(double t, int barcode, double R, double phi) :
		t(t), barcode(barcode), R(R), phi(phi) {};
};

typedef std::vector<Measurment> MeasurmentList;

MeasurmentList ReadMeasurmentList(long time_start_sec, long time_start_msec, double duration_sec, const Barcodes& codes = {});

struct Odometry {
	double t;
	double v, omega;
	Odometry(double t, double v, double omega) :
		t(t), v(v), omega(omega) {};
};

typedef std::vector<Odometry> OdometryData;

OdometryData ReadOdometryData(long time_start_sec, long time_start_msec, double duration_sec);

void ConvertGroundTruthData(long time_start_sec, long time_start_msec, double duration_sec);