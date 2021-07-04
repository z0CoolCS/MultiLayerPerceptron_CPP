#include <iostream>
#include "../../Eigen/Dense"
#include "../Layer.h"

#ifndef _SIGMOIF_H_
#define _SIGMOIF_H_
class Sigmoid : public Layer {
    public:
    Sigmoid () {
        std::cout<<1<<std::endl;
    }

    Eigen::MatrixXd forward (Eigen::MatrixXd input) {
        input = input * -1;
        return 1 / (1 + input.array().exp());
    } 

    Eigen::MatrixXd backward (Eigen::MatrixXd input, Eigen::MatrixXd grad) {
        Eigen::ArrayXd sigmoid = 1 / (1 + input.array().exp());
        Eigen::MatrixXd grad_sigmoid = (sigmoid * (1 - sigmoid)).matrix();
        return grad_sigmoid.array() * grad.array();
    }
};

#endif
