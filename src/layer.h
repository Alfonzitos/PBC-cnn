#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <vector>

#include "helpMath.h"
#include "helpFuncs.h"


class Layer
{
public:
    virtual Tensor &evaluate(Tensor& A) = 0;
    virtual void calculateGradient(Tensor &previousLayerActivation, Tensor &dC_dA__orAnswerV) = 0;

    Tensor dA; // Cost gradient for inputs to layer
    Tensor Z;  // Z = W*X + B
    Tensor A;  // A = RelU(Z) or softMax(Z);

    virtual void updateParams(float learningRate, std::size_t batchSize) = 0;


};

class DenseLayer : public Layer
{
public:
    Eigen::MatrixXf Weights;
    Eigen::VectorXf Bias;

    Eigen::MatrixXf dW; // Cost gradient for Weights
    Eigen::VectorXf dB; // Cost gradient for Bias
    DenseLayer(Eigen::MatrixXf &W, Eigen::VectorXf &B);
    void updateParams(float learningRate, std::size_t batchSize) override;
    void calculateGradient(Tensor &previousLayerActivation, Tensor &dC_dA) override;
    Tensor &evaluate(Tensor &previousActivation) override;
};

class OutputLayer : public Layer
{
public:
    Eigen::MatrixXf Weights;
    Eigen::VectorXf Bias;

    Eigen::MatrixXf dW; // Cost gradient for Weights
    Eigen::VectorXf dB; // Cost gradient for Bias
    OutputLayer(Eigen::MatrixXf &W, Eigen::VectorXf &B);
    void calculateGradient(Tensor &previousLayerActivation, Tensor &answerVector) override;


    void updateParams(float learningRate, std::size_t batchSize) override;

    Tensor &evaluate(Tensor &A) override;
};

class ConvolutionLayer : public Layer
{
public:
    int inputChannels;
    int outputChannels;
    int kernelWidth;
    int kernelHeight;

    int paddingX;
    int paddingY;

    int stride;

    std::vector<Kernel> filters;

    std::vector<Tensor> dFilters;
    Eigen::VectorXf convDB;

    ConvolutionLayer(int inputChannels, int outputChannels,
                     int kernelWidth, int kernelHeight,
                    int paddingX, int paddingY, int stride);

    void initializeParameters();

    void updateParams(float learningRate, std::size_t batchSize) override;

    void calculateGradient(Tensor &previousLayerActivation, Tensor &dC_dA);
    Tensor &evaluate(Tensor &previousLaterActivation);
};

class maxPoolingLayer : public Layer
{
public:
    int windowWidth;
    int windowHeight;

    int stride;

    Tensor mask;
    void updateParams(float, std::size_t) override
    {
    }

    maxPoolingLayer(int windowWidth, int windowHeight, int stride) : windowWidth(windowWidth), windowHeight(windowHeight), stride(stride)
    {
    }

    void calculateGradient(Tensor &previousLayerActivation, Tensor &dC_dA);
    Tensor &evaluate(Tensor &previousLaterActivation) override;
};

class globalAveragePoolingLayer : public Layer
{
public:
    globalAveragePoolingLayer()
    {
    }
    void updateParams(float, std::size_t) override
    {
    }
    Tensor &evaluate(Tensor& previousActivation);
    void calculateGradient(Tensor &previousLayerActivation, Tensor &dC_dA) override;
};