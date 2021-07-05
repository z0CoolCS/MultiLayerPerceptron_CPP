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
        return (1 / (1 + input.array().exp())).matrix();
    } 

    Eigen::MatrixXd backward (Eigen::MatrixXd input, Eigen::MatrixXd grad) {
        input = input * -1;
        input = input.array().exp();
        input = input.array() / (1.0 + input.array()).pow(2);
        return input.array() * grad.array();
    }

};

#endif
