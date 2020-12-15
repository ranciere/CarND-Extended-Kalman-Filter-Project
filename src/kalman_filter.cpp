#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in, MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in)
{
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict(double delta_T)
{
  // Udpate F and Q by noise & delta_T
  double dt_2 = pow(delta_T, 2);
  double dt_3 = pow(delta_T, 3);
  double dt_4 = pow(delta_T, 4);

  // Noise for Q matrix
  double noise_ax = 9.0;
  double noise_ay = 9.0;

  // Modify F according to elapsed time
  F_(0, 2) = delta_T;
  F_(1, 3) = delta_T;

  Q_ = MatrixXd(4, 4);
  Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
      0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
      dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
      0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z)
{
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  x_ = x_ + (K * y);
  const auto x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z)
{
  // Update the state by using Extended Kalman Filter equations
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  double rho = sqrt(pow(px, 2) + pow(py, 2));
  double theta = atan2(py, px);
  double rho_dot = (px * vx + py * vy) / rho;

  VectorXd Hj(3);
  Hj << rho, theta, rho_dot;

  // Normalize y angle
  VectorXd y = z - Hj;
  y(1) -= (2 * M_PI) * floor((y[1] + M_PI) / (2 * M_PI));

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // New estimate state.
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
