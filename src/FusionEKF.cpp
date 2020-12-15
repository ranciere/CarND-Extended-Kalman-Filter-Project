#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

FusionEKF::FusionEKF() : is_initialized_(false), previous_timestamp_(0)
{
  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
      0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
      0, 0.0009, 0,
      0, 0, 0.09;

  //measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
      0, 1, 0, 0;

  //initialize state covariance matrix
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1000, 0,
      0, 0, 0, 1000;

  //initialize initial transition matrix
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
      0, 1, 0, 1,
      0, 0, 1, 0,
      0, 0, 0, 1;
}

void FusionEKF::Init(const MeasurementPackage &measurement_pack)
{
  assert(is_initialized_ == false);
  // first measurement
  std::cout << "EKF: " << std::endl;
  ekf_.x_ = VectorXd::Ones(4);
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    // From polar to cartesian & initialize
    double rho = measurement_pack.raw_measurements_(0);
    double phi = measurement_pack.raw_measurements_(1);
    double rho_dot = measurement_pack.raw_measurements_(2);

    // Coordinates convertion from polar to cartesian
    double px = (rho * cos(phi) >= 0.0001) ? (rho * cos(phi)) : 0.0001;
    double py = (rho * sin(phi) >= 0.0001) ? (rho * sin(phi)) : 0.0001;
    double vx = rho_dot * cos(phi);
    double vy = rho_dot * sin(phi);

    // Push results to Kalman Filter.
    ekf_.x_ << px, py, vx, vy;
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
  {
    /**
      Initialize state.
      */
    ekf_.x_ << measurement_pack.raw_measurements_[0],
        measurement_pack.raw_measurements_[1],
        0,
        0;
  }

  // done initializing, no need to predict or update
  is_initialized_ = true;
  previous_timestamp_ = measurement_pack.timestamp_;
}

void FusionEKF::Predict(const MeasurementPackage &measurement_pack)
{
  // Calculate dt and decendants
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  //
  double dt_2 = pow(dt, 2);
  double dt_3 = pow(dt, 3);
  double dt_4 = pow(dt, 4);

  // Noise for Q matrix
  double noise_ax = 9.0;
  double noise_ay = 9.0;

  // Modify F according to elapsed time
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
      0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
      dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
      0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

  // Update the prediction.
  ekf_.Predict();
}

void FusionEKF::Update(const MeasurementPackage &measurement_pack)
{
  if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
  {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }
  else
  {
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
  // Main loop
  // No initialized? => INIT!
  if (!is_initialized_)
  {
    Init(measurement_pack);
  }
  // Predict & Update forever! <3
  else
  {
    Predict(measurement_pack);
    Update(measurement_pack);
  }
  std::cout << "x_ = " << ekf_.x_ << std::endl;
  std::cout << "P_ = " << ekf_.P_ << std::endl;
}
