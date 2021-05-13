#pragma once
#include <vector>
#include "Eigen/Dense"

void SLAM_state_example();

void SLAM_output_example();

std::vector<std::pair<int, std::pair<double, double>>> SLAM_stateupdate_data();

void to_compute();

void SLAM_outputupdate_data(Eigen::VectorXi& N, Eigen::VectorXi&Na, Eigen::MatrixXd& out_old, Eigen::MatrixXd& out_new);
