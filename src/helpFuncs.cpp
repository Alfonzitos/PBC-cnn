#include "helpFuncs.h"
#include "layer.h"

void createParentDirectory(std::string &path)
{
    std::filesystem::path filePath(path);

    if (!filePath.parent_path().empty())
    {
        std::filesystem::create_directories(filePath.parent_path());
    }
}

void saveMatrix(std::string &path, Eigen::MatrixXf &matrix)
{
    createParentDirectory(path);

    std::ofstream file(path, std::ios::binary);

    file.write(reinterpret_cast<const char *>(matrix.data()), matrix.size() * sizeof(float));
}

void loadMatrix(std::string &path, Eigen::MatrixXf &matrix)
{
    std::ifstream file(path, std::ios::binary);

    file.read(reinterpret_cast<char *>(matrix.data()), matrix.size() * sizeof(float));
}

void saveVector(std::string &path, Eigen::VectorXf &vector)
{
    createParentDirectory(path);

    std::ofstream file(path, std::ios::binary);

    file.write(reinterpret_cast<const char *>(vector.data()), vector.size() * sizeof(float));
}

void loadVector(std::string &path, Eigen::VectorXf &vector)
{
    std::ifstream file(path, std::ios::binary);

    file.read(reinterpret_cast<char *>(vector.data()), vector.size() * sizeof(float));
}

void saveFilters(std::string &path, std::vector<Kernel> &filters)
{
    createParentDirectory(path);

    std::ofstream file(path, std::ios::binary);

    for (Kernel &filter : filters)
    {
        file.write(reinterpret_cast<const char *>(filter.weights.data.data()), filter.weights.data.size() * sizeof(float));
    }
}

void loadFilters(std::string &path, std::vector<Kernel> &filters)
{
    std::ifstream file(path, std::ios::binary);

    for (Kernel &filter : filters)
    {
        file.read(reinterpret_cast<char *>(filter.weights.data.data()), filter.weights.data.size() * sizeof(float));
    }
}

void saveFilterBiases(std::string &path, std::vector<Kernel> &filters)
{
    createParentDirectory(path);

    std::ofstream file(path, std::ios::binary);

    for (Kernel &filter : filters)
    {
        file.write(reinterpret_cast<const char *>(&filter.bias), sizeof(float));
    }
}

void loadFilterBiases(std::string &path, std::vector<Kernel> &filters)
{
    std::ifstream file(path, std::ios::binary);

    for (Kernel &filter : filters)
    {
        file.read(reinterpret_cast<char *>(&filter.bias), sizeof(float));
    }
}

Eigen::VectorXf getAnswerVector(std::string dirName)
{
    Eigen::VectorXf y(8);
    y.setZero();
    static std::map<std::string, Answers> answerMap =
        {
            {"basophil", basophil},
            {"eosinophil", eosinophil},
            {"erythroblast", erythroblast},
            {"ig", ig},
            {"lymphocyte", lymphocyte},
            {"monocyte", monocyte},
            {"neutrophil", neutrophil},
            {"platelet", platelet}};

    y[answerMap.at(dirName)] = 1.0f;
    return y;
}

Eigen::MatrixXf createRandomWeights(int outputCount, int inputCount)
{
    static std::mt19937 generator(std::random_device{}());

    // https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/
    float standardDeviation =
        std::sqrt(2.0f / static_cast<float>(inputCount));

    std::normal_distribution<float> distribution(0.0f, standardDeviation);

    Eigen::MatrixXf weights(outputCount, inputCount);

    for (int row = 0; row < weights.rows(); ++row)
    {
        for (int column = 0; column < weights.cols(); ++column)
        {
            weights(row, column) = distribution(generator);
        }
    }

    return weights;
}

Tensor::Tensor(Eigen::VectorXf vector) : channels(vector.size()),
                                         width(1),
                                         height(1),
                                         data(std::move(vector))
{
}

Tensor::Tensor(int channels, int width, int height) : channels(channels),
                                                      width(width),
                                                      height(height),
                                                      data(channels * width * height)
{
}

int Tensor::index(int channel, int x, int y)
{
    return channel * width * height + y * width + x;
}

float &Tensor::at(int channel, int x, int y)
{
    return data[index(channel, x, y)];
}

void Tensor::resize(int newChannels, int newWidth, int newHeight)
{
    channels = newChannels;
    width = newWidth;
    height = newHeight;
    data.resize(newChannels * newWidth * newHeight);
}

Kernel::Kernel(int channels, int width, int height) : weights(channels, width, height)
{
}

float &Kernel::at(int channel, int x, int y)
{
    return weights.at(channel, x, y);
}