#include "layer.h"

DenseLayer::DenseLayer(Eigen::MatrixXf &W, Eigen::VectorXf &B)
{
    Weights = std::move(W);
    Bias = std::move(B);

    dW = Eigen::MatrixXf::Zero(Weights.rows(), Weights.cols());
    dB = Eigen::VectorXf::Zero(Bias.size());
}

OutputLayer::OutputLayer(Eigen::MatrixXf &W, Eigen::VectorXf &B)
{
    Weights = std::move(W);
    Bias = std::move(B);

    dW = Eigen::MatrixXf::Zero(Weights.rows(), Weights.cols());
    dB = Eigen::VectorXf::Zero(Bias.size());
}

Tensor &DenseLayer::evaluate(Tensor &previousActivation)
{

    Z.resize(Weights.rows(), 1, 1);
    A.resize(Weights.rows(), 1, 1);

    Z.data.noalias() = Weights * previousActivation.data + Bias;
    A.data = Z.data.array().max(0.0f);

    return A;
}

void DenseLayer::calculateGradient(Tensor &previousLayerActivation, Tensor &dC_dA)
{

    // derivate of RelU(Z(li))
    // Eigen::VectorXf relUDeriv = Z.data.unaryExpr(std::function(relUDerivative));
    Eigen::VectorXf delta = dC_dA.data.array() * (A.data.array() > 0.0f).cast<float>();
    // delC/delZ
    // Eigen::VectorXf delta = dC_dA.data.cwiseProduct(relUDeriv);

    // accumulate current layers gradient for weights, bias to be applied once enough training data has been through?

    // https://stackoverflow.com/questions/74199536/computing-the-outer-product-of-two-vectors-in-eigen-c
    dW += delta * previousLayerActivation.data.transpose(); // same as taking the outer product
    dB += delta;

    // define how A(L-N) depends on C to be passed back
    dA.resize(previousLayerActivation.channels, previousLayerActivation.width, previousLayerActivation.height);
    dA.data.noalias() = Weights.transpose() * delta;
}

void DenseLayer::updateParams(float learningRate, std::size_t batchSize)
{
    float scale = learningRate / static_cast<float>(batchSize);

    Weights.noalias() -= scale * dW;
    Bias.noalias() -= scale * dB;

    dW.setZero();
    dB.setZero();
}

Tensor &OutputLayer::evaluate(Tensor &previousActivation /*input to layer*/)
{

    // Feedforward step
    // z = W*A + B
    Z.resize(Weights.rows(), 1, 1);
    A.resize(Weights.rows(), 1, 1);
    Z.data.noalias() = (Weights * previousActivation.data) + Bias;

    // Softmax function, best for multi-class classification, results in a vector of probabilities, whose sum is 1
    float highestElement = Z.data.maxCoeff();

    A.data = (Z.data.array() - highestElement).exp();

    A.data = A.data / A.data.sum();

    return A;
}

void OutputLayer::calculateGradient(Tensor &previousLayerActivation, Tensor &answerVector)
{

    // indirect calculation of derivative of cost
    // Eigen::VectorXf dC_dA = A.data - answerVector.data;
    // Eigen::MatrixXf jacobian =
    //     A.data.asDiagonal().toDenseMatrix() - (A.data * A.data.transpose());
    // Eigen::VectorXf delta = jacobian * dC_dA;

    Eigen::VectorXf delta = A.data - answerVector.data;

    dW += delta * previousLayerActivation.data.transpose();
    dB += delta;

    dA.resize(previousLayerActivation.data.size(), 1, 1);

    dA.data = Weights.transpose() * delta;
}

void OutputLayer::updateParams(float learningRate, std::size_t batchSize)
{
    float scale = learningRate / static_cast<float>(batchSize);

    Weights.noalias() -= scale * dW;
    Bias.noalias() -= scale * dB;

    dW.setZero();
    dB.setZero();
}

ConvolutionLayer::ConvolutionLayer(int inputChannels, int outputChannels,
                                   int kernelWidth, int kernelHeight,
                                   int paddingX, int paddingY, int stride) : inputChannels(inputChannels), outputChannels(outputChannels),
                                                                             kernelWidth(kernelWidth), kernelHeight(kernelHeight),
                                                                             convDB(Eigen::VectorXf::Zero(outputChannels)), paddingX(paddingX),
                                                                             paddingY(paddingY), stride(stride)
{
    for (int d = 0; d < outputChannels; ++d)
    {
        filters.emplace_back(inputChannels, kernelWidth, kernelHeight);
        dFilters.emplace_back(inputChannels, kernelWidth, kernelHeight);
        dFilters.back().data.setZero();
    }
    initializeParameters();
}

Tensor &ConvolutionLayer::evaluate(Tensor &previousLayerActivation)
{

    int inputChannels = previousLayerActivation.channels;
    int inputWidth = previousLayerActivation.width;
    int inputHeight = previousLayerActivation.height;

    // https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
    int outputWidth = (inputWidth + 2 * paddingX - kernelWidth) / stride + 1;

    int outputHeight = (inputHeight + 2 * paddingY - kernelHeight) / stride + 1;

    Z.resize(outputChannels, outputWidth, outputHeight);
    A.resize(outputChannels, outputWidth, outputHeight);

    const float *input = previousLayerActivation.data.data();

    float *zOutput = Z.data.data();
    float *aOutput = A.data.data();

    const int inputPlaneSize = inputWidth * inputHeight;
    const int outputPlaneSize = outputWidth * outputHeight;
    const int kernelPlaneSize = kernelWidth * kernelHeight;

#pragma omp parallel for schedule(static)
    for (int f = 0; f < outputChannels; ++f)
    {
        float *filter = filters[f].weights.data.data();

        float bias = filters[f].bias;

        int outputChannelOffset = f * outputPlaneSize;

        for (int y = 0; y < outputHeight; ++y)
        {
            // find Y for input
            int inputOriginY = y * stride - paddingY;

            // if e.g kernelY = 0 is outside bounds at -1 in input, making kernelY start at = 1
            int kernelYBegin = std::max(0, -inputOriginY);

            // if kernelheight = 3, and there are only two valid rows at||below current y
            // then make end of kernel appear "faster", to not sample outside
            int kernelYEnd = std::min(kernelHeight, inputHeight - inputOriginY);

            int outputRowOffset = outputChannelOffset + y * outputWidth;

            for (int x = 0; x < outputWidth; ++x)
            {
                // find which X we are at in input
                const int inputOriginX = x * stride - paddingX;

                // same as for y but for X
                int kernelXBegin = std::max(0, -inputOriginX);

                int kernelXEnd = std::min(kernelWidth, inputWidth - inputOriginX);

                // init sum for each bias, which we will add each pair of (x,y) from all input channels to
                float sum = bias;

                for (int c = 0; c < inputChannels; ++c)
                {
                    int inputChannelOffset = c * inputPlaneSize;

                    int filterChannelOffset = c * kernelPlaneSize;

                    for (int ky = kernelYBegin; ky < kernelYEnd; ++ky)
                    {
                        int inputY = inputOriginY + ky;

                        int inputRowOffset = inputChannelOffset + inputY * inputWidth + inputOriginX;

                        int filterRowOffset = filterChannelOffset + ky * kernelWidth;

                        for (int kx = kernelXBegin; kx < kernelXEnd; ++kx)
                        {
                            sum += input[inputRowOffset + kx] * filter[filterRowOffset + kx];
                        }
                    }
                }

                int outputIndex = outputRowOffset + x;

                zOutput[outputIndex] = sum;
                aOutput[outputIndex] = std::max(0.0f, sum);
            }
        }
    }

    return A;
}

void ConvolutionLayer::initializeParameters()
{
    static std::mt19937 generator(std::random_device{}());

    int fanIn /*flatten num inputs that affect 1 output*/ = inputChannels * kernelWidth * kernelHeight;

    float standardDeviation = std::sqrt(2.0f / static_cast<float>(fanIn));

    std::normal_distribution<float> distribution(0.0f, standardDeviation);

    for (Kernel &filter : filters)
    {
        for (int i = 0; i < filter.weights.data.size(); ++i)
        {
            filter.weights.data[i] = distribution(generator);
        }

        filter.bias = 0.0f;
    }
}

void ConvolutionLayer::calculateGradient(Tensor &previousLayerActivation, Tensor &dC_dA)
{
    int inputChannels = previousLayerActivation.channels;
    int inputWidth = previousLayerActivation.width;
    int inputHeight = previousLayerActivation.height;
    int inputPlaneSize = inputWidth * inputHeight;
    int inputSize = inputChannels * inputPlaneSize;
    int outputWidth = dC_dA.width;
    int outputHeight = dC_dA.height;
    int outputPlaneSize = outputWidth * outputHeight;
    int kernelPlaneSize = kernelWidth * kernelHeight;

    dA.resize(inputChannels, inputWidth, inputHeight);

    dA.data.setZero();

    int maximumThreadCount = omp_get_max_threads();

    std::vector<Eigen::VectorXf> threadGradients(maximumThreadCount);

    for (Eigen::VectorXf &gradient : threadGradients)
    {
        gradient = Eigen::VectorXf::Zero(inputSize);
    }

#pragma omp parallel
    {
        int threadId = omp_get_thread_num();

        Eigen::VectorXf &localPreviousGradient = threadGradients[threadId];

#pragma omp for schedule(static)
        for (int d = 0; d < outputChannels; ++d)
        {
            Eigen::VectorXf &filterGradient = dFilters[d].data;

            Eigen::VectorXf &filterWeights = filters[d].weights.data;

            int outputChannelOffset = d * outputPlaneSize;

            float biasGradient = 0.0f;

            for (int y = 0; y < outputHeight; ++y)
            {
                int inputOriginY = y * stride - paddingY;

                int kernelYBegin = std::max(0, -inputOriginY);

                int kernelYEnd = std::min(kernelHeight, inputHeight - inputOriginY);

                int outputRowOffset = outputChannelOffset + y * outputWidth;

                for (int x = 0; x < outputWidth; ++x)
                {
                    int outputIndex = outputRowOffset + x;

                    if (A.data[outputIndex] <= 0.0f)
                    {
                        continue;
                    }

                    float gradient = dC_dA.data[outputIndex];

                    biasGradient += gradient;

                    int inputOriginX = x * stride - paddingX;

                    int kernelXBegin = std::max(0, -inputOriginX);

                    int kernelXEnd = std::min(kernelWidth, inputWidth - inputOriginX);

                    for (int k = 0; k < inputChannels; ++k)
                    {
                        int inputChannelOffset = k * inputPlaneSize;

                        int kernelChannelOffset = k * kernelPlaneSize;

                        for (int ky = kernelYBegin; ky < kernelYEnd; ++ky)
                        {
                            int inputY = inputOriginY + ky;

                            int inputRowOffset = inputChannelOffset + inputY * inputWidth + inputOriginX;

                            int kernelRowOffset = kernelChannelOffset + ky * kernelWidth;

                            for (int kx = kernelXBegin; kx < kernelXEnd; ++kx)
                            {
                                int inputIndex = inputRowOffset + kx;

                                int kernelIndex = kernelRowOffset + kx;

                                filterGradient[kernelIndex] += previousLayerActivation.data[inputIndex] * gradient;

                                localPreviousGradient[inputIndex] += filterWeights[kernelIndex] * gradient;
                            }
                        }
                    }
                }
            }

            convDB[d] += biasGradient;
        }
    }

#pragma omp parallel for schedule(static)
    for (int inputIndex = 0; inputIndex < inputSize; ++inputIndex)
    {
        float gradientSum = 0.0f;

        for (int threadIndex = 0; threadIndex < maximumThreadCount; ++threadIndex)
        {
            gradientSum += threadGradients[threadIndex][inputIndex];
        }

        dA.data[inputIndex] = gradientSum;
    }
}

void ConvolutionLayer::updateParams(float learningRate, std::size_t batchSize)
{
    const float scale =
        learningRate / static_cast<float>(batchSize);

    for (int f = 0; f < outputChannels; ++f)
    {
        for (int c = 0; c < inputChannels; ++c)
        {
            for (int ky = 0; ky < kernelHeight; ++ky)
            {
                for (int kx = 0; kx < kernelWidth; ++kx)
                {
                    filters[f].at(c, kx, ky) -= scale * dFilters[f].at(c, kx, ky);
                }
            }
        }

        filters[f].bias -= scale * convDB[f];
    }

    // clear accumulated gradients for the next batch
    for (Tensor &gradientFilter : dFilters)
    {
        gradientFilter.data.setZero();
    }

    convDB.setZero();
}

Tensor &maxPoolingLayer::evaluate(Tensor &previousLayerActivation)
{

    // out = ((in + 2*P - K/)S) + 1
    // where P is padding, S is stride and K is window size
    int outputWidth = ((previousLayerActivation.width - windowWidth) / stride) + 1;
    int outputHeight = ((previousLayerActivation.height - windowHeight) / stride) + 1;

    A.resize(previousLayerActivation.channels, outputWidth, outputHeight);

    // initialize mask to zero
    mask.resize(previousLayerActivation.channels, previousLayerActivation.width, previousLayerActivation.height);
    mask.data.setZero();

    for (int c = 0; c < previousLayerActivation.channels; c++)
    {
        for (int x = 0; x < outputWidth; x++)
        {
            for (int y = 0; y < outputHeight; y++)
            {
                float max = -10000000; // arbitrary value that can't be "normally found" as lowest?

                // saved these to know which exact pixel was responsible for value during backprop
                int largestX;
                int largestY;

                int inputX = x * stride;
                int inputY = y * stride;

                // for each position in the input
                for (int i = 0; i < windowWidth; i++)
                {
                    for (int j = 0; j < windowHeight; j++)
                    {
                        float currValue = previousLayerActivation.at(c, inputX + i, inputY + j);
                        if (currValue > max)
                        {
                            max = currValue;
                            largestX = inputX + i;
                            largestY = inputY + j;
                        }
                    }
                }
                // index into output map
                A.at(c, x, y) = max;
                mask.at(c, largestX, largestY) = 1.0f;
            }
        }
    }

    return A;
}

void maxPoolingLayer::calculateGradient(Tensor &previousLayerActivation, Tensor &dC_dA)
{

    // this kind of assumes during forwards pass there were no overlap of the
    // window, leading to "weird" masks. Using a stride of 2 and windows of size 2 makes this a nonissue?
    Tensor delta(dC_dA.channels, dC_dA.width, dC_dA.height);
    delta.data = dC_dA.data;

    dA.resize(previousLayerActivation.channels, previousLayerActivation.width, previousLayerActivation.height);
    dA.data.setZero();

    int outputWidth = ((previousLayerActivation.width - windowWidth) / stride) + 1;
    int outputHeight = ((previousLayerActivation.height - windowHeight) / stride) + 1;

    for (int c = 0; c < previousLayerActivation.channels; c++)
    {
        for (int x = 0; x < outputWidth; x++)
        {
            for (int y = 0; y < outputHeight; y++)
            {
                // for each pixel

                for (int i = 0; i < windowWidth; i++)
                {
                    for (int j = 0; j < windowHeight; j++)
                    {
                        int inputX = x * stride + i;
                        int inputY = y * stride + j;
                        if (mask.at(c, inputX, inputY) == 1.0f)
                        {
                            dA.at(c, inputX, inputY) += dC_dA.at(c, x, y);
                        }
                    }
                }
            }
        }
    }
}

Tensor &globalAveragePoolingLayer::evaluate(Tensor &previousActivation)
{
    A.resize(previousActivation.channels, 1, 1);

    int filterSize = previousActivation.height * previousActivation.width;
    for (int c = 0; c < previousActivation.channels; c++)
    {
        int offset = c * filterSize;
        A.data[c] = previousActivation.data.segment(offset, filterSize).mean();
    }
    return A;
}

void globalAveragePoolingLayer::calculateGradient(Tensor &previousActivation, Tensor &dC_dA)
{
    dA.resize(previousActivation.channels, previousActivation.width, previousActivation.height);

    int spatialSize = previousActivation.width * previousActivation.height;

    for (int c = 0; c < previousActivation.channels; ++c)
    {
        float gradient = dC_dA.data[c] / (spatialSize);

        dA.data.segment(c * spatialSize, spatialSize).setConstant(gradient);
    }
}