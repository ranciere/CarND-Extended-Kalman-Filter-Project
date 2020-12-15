#include <iostream>
#include <cassert>
#include <cmath>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

Tools::Tools() {}

VectorXd Tools::CalculateRMSE(const std::vector<VectorXd> &estimations, const std::vector<VectorXd> &ground_truth)
{
   // Temporary and result vector
   VectorXd rmse = VectorXd::Zero(4);
   // Check for divide by zero error
   assert(!estimations.empty());
   assert(estimations.size() == ground_truth.size());
   // Calculate squared residuals.
   for (unsigned int i = 0; i < estimations.size(); i++)
   {
      VectorXd residual = estimations[i] - ground_truth[i];
      residual = residual.cwiseProduct(residual);
      rmse += residual;
   }
   // Calculate mean
   rmse = rmse / estimations.size();
   // Calculate squared root
   rmse = rmse.cwiseSqrt();

   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{
   // Get variables from matrix
   MatrixXd Hj(3, 4);
   double px = x_state(0);
   double py = x_state(1);
   double vx = x_state(2);
   double vy = x_state(3);

   // Pre-compute constants for human readability
   double den = pow(px, 2) + pow(py, 2);
   double den_sqrt = sqrt(den);
   double den_pow = pow(den, 1.5);

   // Check for division by zero.
   assert(fabs(den) > 0.0001);

   // Compute the Jacobian Matrix.
   Hj << (px / den_sqrt), (py / den_sqrt), 0, 0,
       -(py / den), (px / den), 0, 0,
       py * (vx * py - vy * px) / den_pow, px * (px * vy - py * vx) / den_pow, px / den_sqrt, py / den_sqrt;

   return Hj;
}
