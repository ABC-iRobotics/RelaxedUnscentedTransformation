#include "readData.h"
#include <fstream>
#include <string>
#include <algorithm>

Barcodes readBarcodes() {
	Barcodes out;
	// File pointer 
	std::fstream fin;
	// Open an existing file 
	fin.open(std::string(DATASET_PATH) + "SLAM_dataset/Barcodes.dat", std::ios::in);
	if (!fin)
		std::logic_error("File not found");
	// Read first 4 comment rows
	{
		std::string word;
		getline(fin, word, '\n');
		getline(fin, word, '\n');
		getline(fin, word, '\n');
		getline(fin, word, '\n');
	}
	// Get Barcodes
	{
		//first five are robots
		std::string word;
		for (int i = 0; i < 5; i++)
			getline(fin, word, '\n');
	}
	{
		while (!fin.eof()) {
			std::string word;
			int n = -1;
			while (n == -1) {
				try {
					getline(fin, word, ' ');
					while (word.size() == 0 || word[0] == ' ' || word[0] == '\t')
						getline(fin, word, ' ');
					n = stoi(word);
				}
				catch (...) {
					n = -1;
				}
			}
			n = stoi(word);
			//std::cout << "n: " << n << std::endl;
			int barcode = -1;
			while (barcode == -1) {
				try {
					getline(fin, word, ' ');
					while (word.size() == 0 || word[0] == ' ' || word[0] == '\t')
						getline(fin, word, ' ');
					barcode = stoi(word);
				}
				catch (...) {
					barcode = -1;
				}
			}	
			//std::cout << "barcode: " << barcode << std::endl;
			getline(fin, word, '\n');
			out.insert({ n,barcode });
		}
	}
	fin.close();
	return out;
}

bool doesContain(const Barcodes& codes, int value) {
	return std::find_if(codes.begin(), codes.end(), [value](const auto& mo) {return mo.second == value; }) != codes.end();
}

MeasurmentList ReadMeasurmentList(long time_start_sec, long time_start_msec, double duration_sec, const Barcodes& codes) {
	MeasurmentList out;
	// File pointer 
	std::fstream fin;
	// Open an existing file 
	fin.open(std::string(DATASET_PATH) + "SLAM_dataset/Robot1_Measurement.dat", std::ios::in);
	if (!fin)
		std::logic_error("File not found");
	// Read first 4 comment rows
	{
		std::string word;
		getline(fin, word, '\n');
		getline(fin, word, '\n');
		getline(fin, word, '\n');
		getline(fin, word, '\n');
	}
	// Get data
	{
		while (!fin.eof()) {
			std::string word;
			// t0
			long t0 = -1;
			while (t0 == -1) {
				try {
					getline(fin, word, '.');
					while (word.size() == 0 || word[0] == ' ' || word[0] == '\t')
						getline(fin, word, ' ');
					t0 = stol(word);
				}
				catch (...) {
					t0 = -1;
				}
			}
			// t1
			getline(fin, word, ' ');
			double t1 = double(stol(word)) / pow(10,word.length());
			// barcode
			int barcode = -1;
			while (barcode == -1) {
				try {
					getline(fin, word, ' ');
					while (word.size() == 0 || word[0] == ' ' || word[0] == '\t')
						getline(fin, word, ' ');
					barcode = stol(word);
				}
				catch (...) {
					barcode = -1;
				}
			}
			// R
			double R = -1;
			while (R == -1) {
				try {
					getline(fin, word, ' ');
					while (word.size() == 0 || word[0] == ' ' || word[0] == '\t')
						getline(fin, word, ' ');
					R = stod(word);
				}
				catch (...) {
					R = -1;
				}
			}
			// phi
			double phi  = -1;
			while (phi == -1) {
				try {
					getline(fin, word, ' ');
					while (word.size() == 0 || word[0] == ' ' || word[0] == '\t')
						getline(fin, word, ' ');
					phi = stod(word);
				}
				catch (...) {
					phi = -1;
				}
			}
			getline(fin, word, '\n');
			t0 -= time_start_sec;
			t1 -= double(time_start_msec) / 1000.;
			double t = t0 + t1;
			if (t>=0 && t< duration_sec && doesContain(codes, barcode))
				out.push_back({ t,barcode,R,phi });
		}
	}
	fin.close();
	return out;
}

OdometryData ReadOdometryData(long time_start_sec, long time_start_msec, double duration_sec) {
	OdometryData out;
	// File pointer 
	std::fstream fin;
	// Open an existing file 
	fin.open(std::string(DATASET_PATH) + "SLAM_dataset/Robot1_Odometry.dat", std::ios::in);
	if (!fin)
		std::logic_error("File not found");
	// Read first 4 comment rows
	{
		std::string word;
		getline(fin, word, '\n');
		getline(fin, word, '\n');
		getline(fin, word, '\n');
		getline(fin, word, '\n');
	}
	// Get data
	{
		while (!fin.eof()) {
			std::string word;
			// t0
			long t0 = -1;
			while (t0 == -1) {
				try {
					getline(fin, word, '.');
					t0 = stol(word);
				}
				catch (...) {
					t0 = -1;
				}
			}
			// t1
			getline(fin, word, ' ');
			double t1 = double(stol(word)) / pow(10, word.length());
			// v
			double v = -1;
			while (v == -1) {
				try {
					getline(fin, word, ' ');
					while (word.size()==0 || word[0]==' ' || word[0] == '\t')
						getline(fin, word, ' ');
					v = std::stod(word);
				}
				catch (...) {
					v = -1;
				}
			}
			// R
			double omega = -1;
			while (omega == -1) {
				try {
					getline(fin, word, ' ');
					while (word.size() == 0 || word[0] == ' ' || word[0] == '\t')
						getline(fin, word, ' ');
					omega = stod(word);
				}
				catch (...) {
					omega = -1;
				}
			}
			getline(fin, word, '\n');
			t0 -= time_start_sec;
			t1 -= double(time_start_msec) / 1000.;
			double t = t0 + t1;
			if (t >= 0 && t < duration_sec)
				out.push_back({ t,v,omega });
				
		}
	}
	fin.close();
	return out;
}

void ConvertGroundTruthData(long time_start_sec, long time_start_msec, double duration_sec) {
	// open output file
	std::fstream fout;
	fout.open(std::string(DATASET_PATH) + "groundtruth.m", std::ios::out | std::ios::trunc);
	// write path into the output file
	{
		// Open an existing file 
		std::fstream fin;
		fin.open(std::string(DATASET_PATH) + "SLAM_dataset/Robot1_Groundtruth.dat", std::ios::in);
		if (!fin)
			std::logic_error("File not found");
		// Read first 4 comment rows
		{
			std::string word;
			getline(fin, word, '\n');
			getline(fin, word, '\n');
			getline(fin, word, '\n');
			getline(fin, word, '\n');
		}
		// Get data
		{
			fout << "path_groundtruth_t_x_y_phi = [ ";
			bool firstline = true;
			while (!fin.eof()) {
				std::string word;
				// t0
				long t0 = -1;
				while (t0 == -1) {
					try {
						getline(fin, word, '.');
						t0 = stol(word);
					}
					catch (...) {
						t0 = -1;
					}
				}
				// t1
				getline(fin, word, ' ');
				double t1 = double(stol(word)) / pow(10, word.length());

				double vk[3];
				// x,y,phi
				for (int i = 0; i < 3; i++) {
					double v = -1;
					while (v == -1) {
						try {
							getline(fin, word, ' ');
							while (word.size() == 0 || word[0] == ' ' || word[0] == '\t')
								getline(fin, word, ' ');
							v = std::stod(word);
						}
						catch (...) {
							v = -1;
						}
					}
					vk[i] = v;
				}
				getline(fin, word, '\n');
				t0 -= time_start_sec;
				t1 -= double(time_start_msec) / 1000.;
				double t = t0 + t1;
				if (t >= 0 && t < duration_sec) {
					if (!firstline)
						fout << "; ";
					else
						firstline = false;
					fout << t << " ";
					for (int i = 0; i < 3; i++)
						fout << vk[i] << " ";
					fout << "...\n";
				}
			}
			fout << "];\n";
		}
		fin.close();
	}
	// Write landmark ground truth into the file
	{
		// Open an existing file 
		std::fstream fin;
		fin.open(std::string(DATASET_PATH) + "SLAM_dataset/Landmark_Groundtruth.dat", std::ios::in);
		if (!fin)
			std::logic_error("File not found");
		// Read first 4 comment rows
		{
			std::string word;
			getline(fin, word, '\n');
			getline(fin, word, '\n');
			getline(fin, word, '\n');
			getline(fin, word, '\n');
		}
		// Get data
		{
			int n = 1;
			while (!fin.eof()) {
				std::string word;
				// t0
				int ID = -1;
				while (ID == -1) {
					getline(fin, word, ' ');
					try {
						ID = stol(word);
					}
					catch (...) {
						ID = -1;
					}
				}
				fout << "ID{" << n << "} = " << ID << ";\n";
				// x,y
				for (int j = 0; j < 2; j++) {
					if (j == 0)
						fout << "r{" << n << "} = [ ";
					if (j == 1)
						fout << "Sr{" << n << "} = [ ";

					for (int i = 0; i < 2; i++) {
						double v = -1;
						while (v == -1) {
							try {
								getline(fin, word, ' ');
								while (word.size() == 0 || word[0] == ' ' || word[0] == '\t')
									getline(fin, word, ' ');
								v = std::stod(word);
							}
							catch (...) {
								v = -1;
							}
						}
						fout << v << " ";
					}
					fout << "];\n";
				}
				getline(fin, word, '\n');
				n++;
			}
		}
		fin.close();
	}
	fout.close();
}