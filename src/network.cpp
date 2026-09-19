#include "network.h"
#include <limits>

static std::array<std::string, CLASS_COUNT> CLASS_NAMES =
    {
        "basophil",
        "eosinophil",
        "erythroblast",
        "ig",
        "lymphocyte",
        "monocyte",
        "neutrophil",
        "platelet"};

EvaluationResult Network::evaluateDataset(std::vector<trainingInstance> &dataset)
{
    EvaluationResult result;

    if (dataset.empty())
    {
        return result;
    }

    float totalCost = 0.0f;

    for (trainingInstance &instance : dataset)
    {
        Tensor &output = classifyImage(instance.data);

        Tensor answer(getAnswerVector(instance.dataType));

        totalCost += cost(output, answer);

        Eigen::Index predictedClass;
        output.data.maxCoeff(&predictedClass);

        Eigen::Index correctClass;
        answer.data.maxCoeff(&correctClass);

        std::size_t actualIndex = static_cast<std::size_t>(correctClass);

        std::size_t predictedIndex = static_cast<std::size_t>(predictedClass);

        // Row = actual class
        // Column = predicted class
        ++result.confusion[actualIndex][predictedIndex];

        if (predictedClass == correctClass)
        {
            ++result.correctPredictions;
        }
    }

    result.sampleCount = dataset.size();

    result.averageCost = totalCost / static_cast<float>(result.sampleCount);

    result.accuracy = static_cast<float>(result.correctPredictions) / static_cast<float>(result.sampleCount);

    return result;
}

void Network::trainNetwork(std::vector<trainingInstance> &trainingData, int trainingRuns, int batchSize, std::string configPath, volatile std::sig_atomic_t &stopRequested)
{

    std::string valPath = "C:\\Users\\Alfons\\Desktop\\Biology\\ML\\PBC\\validation";
    Data validation(valPath, 360, 363);

    float bestValidationCost = std::numeric_limits<float>::infinity();
    EvaluationResult initialValidation = evaluateDataset(validation.data);
    bestValidationCost = initialValidation.averageCost;

    std::cout << "Loaded model validation cost: " << initialValidation.averageCost << ", validation accuracy: " << initialValidation.accuracy * 100.0f << "%" << std::endl;

    auto rng = std::default_random_engine{};
    for (int epoch = 0; epoch < trainingRuns; epoch++)
    {

        std::shuffle(std::begin(trainingData), std::end(trainingData), rng);
        float totalCost = 0;

        float rollingCost = 0.0f;
        int rollingSamples = 0;
        auto rollingStartTime = std::chrono::steady_clock::now();
        constexpr int reportEvery = 32;
        // for each batch
        for (int batchStart = 0; batchStart < trainingData.size(); batchStart += batchSize)
        {
            std::chrono::steady_clock::time_point batchStartTime = std::chrono::steady_clock::now();

            int samplesInBatch = 0;
            float batchCost = 0;

            // std::chrono::steady_clock::time_point beforeBatch = //std::chrono::steady_clock::now();

            for (int i = batchStart; i < batchStart + batchSize && i < trainingData.size(); i++)
            {
                // std::chrono::steady_clock::time_point beforeEntireImage = //std::chrono::steady_clock::now();

                trainingInstance &inst = trainingData[i];

                Tensor &inputData = inst.data;

                // std::chrono::steady_clock::time_point beforeClassify = //std::chrono::steady_clock::now();
                Tensor &res = classifyImage(inputData);
                // std::chrono::steady_clock::time_point afterClassify = //std::chrono::steady_clock::now();
                //  std::cout << "ClassifyImage: = " << //std::chrono::duration_cast<//std::chrono::milliseconds>(afterClassify - beforeClassify).count() << "[milliseconds]" << std::endl;

                Tensor answer = Tensor(getAnswerVector(inst.dataType));
                float c = cost(res, answer);
                totalCost += c;
                batchCost += c;

                // std::chrono::steady_clock::time_point beforeBackprop = //std::chrono::steady_clock::now();
                backPropagate(inputData, answer);
                // std::chrono::steady_clock::time_point afterBackprop = //std::chrono::steady_clock::now();
                //  std::cout << "Backprop: = " << //std::chrono::duration_cast<//std::chrono::milliseconds>(afterBackprop - beforeBackprop).count() << "[milliseconds]" << std::endl;

                samplesInBatch += 1;
                // std::chrono::steady_clock::time_point afterEntireImage = //std::chrono::steady_clock::now();
                //  std::cout << "One Image: = " << //std::chrono::duration_cast<//std::chrono::milliseconds>(afterEntireImage - beforeEntireImage).count() << "[milliseconds]" << std::endl;
            }
            // std::chrono::steady_clock::time_point afterBatch = //std::chrono::steady_clock::now();
            //  std::cout << "Batch: = " << //std::chrono::duration_cast<//std::chrono::milliseconds>(afterBatch - beforeBatch).count() << "[milliseconds]" << std::endl;

            for (std::unique_ptr<Layer> &layer : layers)
            {
                // layer->updateParams(0.1, samplesInBatch);
                // layer->updateParams(0.001, samplesInBatch);
                layer->updateParams(0.0003, samplesInBatch);
            }
            if (stopRequested != 0)
            {
                // std::cout << "Saving network due to termination" << std::endl;
                // saveNetwork(configPath);
                // std::cout << "Network saved successfully" << std::endl;
                // return;

                std::cout << "Training stopped." << std::endl;

                return;
            }

            const float meanBatchCost = batchCost / static_cast<float>(samplesInBatch);

            rollingCost += meanBatchCost;
            rollingSamples++;

            if (rollingSamples == reportEvery)
            {
                auto rollingEndTime = std::chrono::steady_clock::now();

                auto rollingTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(rollingEndTime - rollingStartTime).count();

                auto now = std::chrono::system_clock::now();
                std::time_t t = std::chrono::system_clock::to_time_t(now);

                std::ofstream logFile("results.txt", std::ios::app);

                logFile
                    << "Epoch " << epoch
                    << ", batches "
                    << batchStart / batchSize - reportEvery + 2
                    << "-"
                    << batchStart / batchSize + 1
                    << ", rolling cost: "
                    << rollingCost /
                           static_cast<float>(rollingSamples)
                    << ", Seconds elapsed " << rollingTimeMs / 1000 << " at: " << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S") << "\n";

                rollingCost = 0.0f;
                rollingSamples = 0;
                rollingStartTime = std::chrono::steady_clock::now();
            }
            // std::cout << "Batch: " << (batchStart / batchSize + 1) << "; Cost: " << batchCost / samplesInBatch << std::endl;
        }
        const float averageTrainingCost = totalCost / static_cast<float>(trainingData.size());

        const EvaluationResult validationResult = evaluateDataset(validation.data);

        std::ofstream validationLog("results.txt", std::ios::app);

        if (validationResult.averageCost < bestValidationCost)
        {
            bestValidationCost = validationResult.averageCost;

            saveNetwork(configPath);

            validationLog << "Saved new best model with validation cost " << bestValidationCost << std::endl;
        }

        validationLog << "Epoch " << epoch << ", training cost: " << averageTrainingCost << ", validation cost: " << validationResult.averageCost << ", validation accuracy: " << validationResult.accuracy * 100.0f << "%" << " (" << validationResult.correctPredictions << "/" << validationResult.sampleCount << ")" << '\n';

        validationLog << "Per-class validation accuracy:" << std::endl;

        for (std::size_t actualClass = 0; actualClass < CLASS_COUNT; ++actualClass)
        {
            std::size_t classTotal = 0;

            for (std::size_t predictedClass = 0; predictedClass < CLASS_COUNT; ++predictedClass)
            {
                classTotal += validationResult.confusion[actualClass][predictedClass];
            }

            const std::size_t classCorrect = validationResult.confusion[actualClass][actualClass];

            validationLog << "  " << CLASS_NAMES[actualClass] << ": ";

            if (classTotal == 0)
            {
                validationLog << "no validation samples" << std::endl;
                continue;
            }

            const float classAccuracy = static_cast<float>(classCorrect) / static_cast<float>(classTotal);

            validationLog << classAccuracy * 100.0f << "%" << " (" << classCorrect << "/" << classTotal << ")" << std::endl;
        }

        validationLog << "Confusion matrix:" << std::endl
                      << "Rows = actual, columns = predicted" << std::endl;

        validationLog << "actual/predicted";

        for (const std::string &className : CLASS_NAMES)
        {
            validationLog << '    ' << className;
        }

        validationLog << std::endl;

        for (std::size_t actualClass = 0; actualClass < CLASS_COUNT; ++actualClass)
        {
            validationLog << CLASS_NAMES[actualClass];

            for (std::size_t predictedClass = 0; predictedClass < CLASS_COUNT; ++predictedClass)
            {
                validationLog << '    ' << validationResult.confusion[actualClass][predictedClass];
            }

            validationLog << std::endl;
        }

        validationLog << std::endl
                      << std::flush;
    }
}

void Network::backPropagate(Tensor &inputData, Tensor &correctAnswer)
{

    // iterate through all layers and backprop
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        // std::chrono::steady_clock::time_point layerStart = std::chrono::steady_clock::now();

        // if last layer
        if (i == layers.size() - 1)
        {
            layers[i]->calculateGradient(layers[i - 1]->A, correctAnswer);
        }
        // if first layer
        else if (i == 0)
        {
            layers[i]->calculateGradient(inputData, layers[i + 1]->dA);
        }
        // if any midle layer
        else
        {
            layers[i]->calculateGradient(layers[i - 1]->A, layers[i + 1]->dA);
        }

        // std::chrono::steady_clock::time_point layerEnd = std::chrono::steady_clock::now();

        // if (dynamic_cast<maxPoolingLayer *>(layers[i].get()))
        // {
        //     std::cout << "Maxpool backprop: = " << std::chrono::duration_cast<std::chrono::milliseconds>(layerEnd - layerStart).count() << " [milliseconds]" << std::endl;
        // }
        // else if (dynamic_cast<ConvolutionLayer *>(layers[i].get()))
        // {
        //     std::cout << "Convolutional backprop: = "
        //               << std::chrono::duration_cast<std::chrono::milliseconds>(layerEnd - layerStart).count() << " [milliseconds]" << std::endl;
        // }
        // else if (dynamic_cast<globalAveragePoolingLayer *>(layers[i].get()))
        // {
        //     std::cout << "Global average backprop: = "
        //               << std::chrono::duration_cast<std::chrono::milliseconds>(layerEnd - layerStart).count() << " [milliseconds]" << std::endl;
        // }
        // else if (dynamic_cast<OutputLayer *>(layers[i].get()))
        // {
        //     std::cout << "Output backprop: = "
        //               << std::chrono::duration_cast<std::chrono::milliseconds>(layerEnd - layerStart).count() << " [milliseconds]" << std::endl;
        // }
        // else if (dynamic_cast<DenseLayer *>(layers[i].get()))
        // {
        //     std::cout << "Dense backprop: = "
        //               << std::chrono::duration_cast<std::chrono::milliseconds>(layerEnd - layerStart).count() << " [milliseconds]" << std::endl;
        // }
    }
}

Tensor &Network::classifyImage(Tensor &inputVector)
{
    Tensor *current = &inputVector;

    int i = 0;
    for (std::unique_ptr<Layer> &layer : layers)
    {
        // std::chrono::steady_clock::time_point layerStart = std::chrono::steady_clock::now();

        current = &layer->evaluate(*current);
        // std::chrono::steady_clock::time_point layerEnd = std::chrono::steady_clock::now();

        // if (dynamic_cast<maxPoolingLayer *>(layer.get()))
        // {
        //     std::cout << "Maxpool eval: = " << std::chrono::duration_cast<std::chrono::milliseconds>(layerEnd - layerStart).count() << "[milliseconds]" << std::endl;
        // }
        // else if (dynamic_cast<globalAveragePoolingLayer *>(layer.get()))
        // {
        //     std::cout << "GlobalAverage eval: = " << std::chrono::duration_cast<std::chrono::milliseconds>(layerEnd - layerStart).count() << "[milliseconds]" << std::endl;
        // }
        // else if (dynamic_cast<ConvolutionLayer *>(layer.get()))
        // {
        //     std::cout << "Convolutional eval: = " << std::chrono::duration_cast<std::chrono::milliseconds>(layerEnd - layerStart).count() << "[milliseconds]" << std::endl;
        // }
        // else if (dynamic_cast<OutputLayer *>(layer.get()))
        // {
        //     std::cout << "Output eval: = " << std::chrono::duration_cast<std::chrono::milliseconds>(layerEnd - layerStart).count() << "[milliseconds]" << std::endl;
        // }
        // else if (dynamic_cast<DenseLayer *>(layer.get()))
        // {
        //     std::cout << "Dense eval: = " << std::chrono::duration_cast<std::chrono::milliseconds>(layerEnd - layerStart).count() << "[milliseconds]" << std::endl;
        // }
        i++;
    }
    // std::cout << "Output: " << layers.back()->A.data.transpose() << '\n';
    return *current;
}

float Network::cost(Tensor &outputVector, Tensor &answerVector)
{
    float epsilon = 1e-7f;
    return -(answerVector.data.array() * outputVector.data.array().max(epsilon).log()).sum();
    // Eigen::VectorXf temp = outputVector.data - answerVector.data;
    // temp = temp.array().square();

    // return temp.sum() / 2;
}

void Network::saveNetwork(std::string &configPath)
{
    json config;
    std::ifstream(configPath) >> config;

    int i = 0;

    for (auto layerConfig : config["Layers"])
    {
        std::string type = layerConfig["LayerType"];

        if (type == "Convolutional")
        {
            auto *layer = dynamic_cast<ConvolutionLayer *>(layers[i].get());

            std::string filterPath = layerConfig["FilterPath"];
            std::string biasPath = layerConfig["BiasPath"];
            saveFilters(filterPath, layer->filters);

            saveFilterBiases(biasPath, layer->filters);
        }
        else if (type == "Dense")
        {
            std::string weightsPath = layerConfig["WeightsPath"];
            std::string biasPath = layerConfig["BiasPath"];
            auto *layer = dynamic_cast<DenseLayer *>(layers[i].get());

            saveMatrix(weightsPath, layer->Weights);

            saveVector(biasPath, layer->Bias);
        }
        else if (type == "Output")
        {
            std::string weightsPath = layerConfig["WeightsPath"];
            std::string biasPath = layerConfig["BiasPath"];
            auto *layer = dynamic_cast<OutputLayer *>(layers[i].get());

            saveMatrix(weightsPath, layer->Weights);

            saveVector(biasPath, layer->Bias);
        }
        ++i;
    }
}

Network::Network(std::string &filePath)
{

    json networkConf;
    std::ifstream confFile(filePath);

    confFile >> networkConf;

    int currChannels = networkConf["InitialInputSize"]["c"];
    int currX = networkConf["InitialInputSize"]["x"];
    int currY = networkConf["InitialInputSize"]["y"];

    bool isDense = false;
    int currFeatures = 0;

    for (auto layer : networkConf["Layers"])
    {
        if (layer["LayerType"] == "Convolutional")
        {
            // [(W-K+2P)/S]+1
            // calculate output dims
            int paddingX = layer["PaddingX"];
            int paddingY = layer["PaddingY"];
            int stride = layer["Stride"];
            auto l = std::make_unique<ConvolutionLayer>(currChannels, layer["FilterCount"], layer["FilterSize"]["x"], layer["FilterSize"]["y"], paddingX, paddingY, stride);

            std::string filterPath = layer["FilterPath"];
            std::string biasPath = layer["BiasPath"];
            if (std::filesystem::exists(filterPath) &&
                std::filesystem::exists(biasPath))
            {
                loadFilters(filterPath, l->filters);
                loadFilterBiases(biasPath, l->filters);
            }

            layers.push_back(std::move(l));

            currChannels = layer["FilterCount"];
            currX = ((currX - layer["FilterSize"]["x"] + (2 * paddingX)) / stride) + 1;
            currY = ((currY - layer["FilterSize"]["y"] + (2 * paddingY)) / stride) + 1;
        }
        else if (layer["LayerType"] == "Max Pooling")
        {
            // calculate output dims
            currChannels = currChannels; // unchanged amount of filters
            int stride = layer["Stride"];

            currX = ((currX - layer["WindowSize"]["x"]) / stride) + 1;
            currY = ((currY - layer["WindowSize"]["y"]) / stride) + 1;

            auto l = std::make_unique<maxPoolingLayer>(layer["WindowSize"]["x"], layer["WindowSize"]["y"], stride);
            layers.push_back(std::move(l));
        }
        else if (layer["LayerType"] == "Global Average Pooling")
        {
            currX = 1;
            currY = 1;
            auto l = std::make_unique<globalAveragePoolingLayer>();
            layers.push_back(std::move(l));
        }

        else if (layer["LayerType"] == "Dense")
        {
            // calculate output dims
            if (!isDense)
            {
                isDense = true;
                currFeatures = currChannels * currX * currY;
            }

            Eigen::MatrixXf w = createRandomWeights(layer["numNeurons"], currFeatures);

            int numNeurons = layer["numNeurons"];
            Eigen::VectorXf b(numNeurons);
            b.setZero();
            std::string weightPath = layer["WeightsPath"];
            std::string biasPath = layer["BiasPath"];

            if (std::filesystem::exists(weightPath) && std::filesystem::exists(biasPath))
            {
                loadMatrix(weightPath, w);
                loadVector(biasPath, b);
            }
            currFeatures = layer["numNeurons"];

            auto l = std::make_unique<DenseLayer>(std::move(w), std::move(b));
            layers.push_back(std::move(l));
        }
        else if (layer["LayerType"] == "Output")
        {
            // calculate output dims
            if (!isDense)
            {
                isDense = true;
                currFeatures = currChannels * currX * currY;
            }

            Eigen::MatrixXf w = createRandomWeights(layer["numNeurons"], currFeatures);
            int numNeurons = layer["numNeurons"];
            Eigen::VectorXf b(numNeurons);
            b.setZero();

            std::string weightPath = layer["WeightsPath"];
            std::string biasPath = layer["BiasPath"];

            if (std::filesystem::exists(weightPath) && std::filesystem::exists(biasPath))
            {
                loadMatrix(weightPath, w);
                loadVector(biasPath, b);
            }

            currFeatures = layer["numNeurons"];

            auto l = std::make_unique<OutputLayer>(w, b);
            layers.push_back(std::move(l));
        }
    }
}

void Network::classifyFile(const std::string &imagePath)
{
    std::string path = imagePath;

    cv::Mat image = Data::readFile(path, 360, 363);

    Tensor input = matToTensor(image);

    Tensor &output = classifyImage(input);

    Eigen::Index predictedClass;
    const float confidence = output.data.maxCoeff(&predictedClass);

    // dec not scientific
    std::cout << std::fixed << std::setprecision(2);

    for (std::size_t i = 0; i < CLASS_NAMES.size(); ++i)
    {
        std::cout << CLASS_NAMES[i] << ": " << output.data[static_cast<Eigen::Index>(i)] * 100.0f << "%" << std::endl;
    }

    std::cout << std::endl << "Prediction: " << CLASS_NAMES[static_cast<std::size_t>(predictedClass)] << " (" << confidence * 100.0f << "%)" << std::endl;
}