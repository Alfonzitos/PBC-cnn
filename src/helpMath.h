#pragma once


#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include "helpFuncs.h"


// https://libeigen.gitlab.io/eigen/docs-5.0/classEigen_1_1DenseBase.html#ae7d61ba5a2bdff148fc072c91c23bed9

float relUDerivative(float element);

Eigen::MatrixXf softMaxDerivative(Eigen::VectorXf &inputVector /*softmax outputvector, i.e last layer activation*/);
