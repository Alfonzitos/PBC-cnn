#include "helpFuncs.h"



float relUDerivative(float element)
{
    if (element > 0)
    {
        return 1;  
    }
    else
    {
        return 0;
    }
}


Eigen::MatrixXf softMaxDerivative(Eigen::VectorXf &inputVector /*softmax outputvector, i.e last layer activation*/)
{
    Eigen::MatrixXf inputVectorAsDiagonalMatrix = inputVector.asDiagonal(); 
    Eigen::MatrixXf aTimesaTransposed = inputVector * inputVector.transpose();

    Eigen::MatrixXf delAdivdelZ = inputVectorAsDiagonalMatrix - aTimesaTransposed;
    return delAdivdelZ;
}
