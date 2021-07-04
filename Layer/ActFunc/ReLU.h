#include <iostream>
#include "../../Eigen/Dense"
#include "../Layer.h"

#ifndef _RELU_H_
#define _RELU_H_
class ReLU : public Layer {
    public:
    ReLU () {
    }

    Eigen::MatrixXd forward (Eigen::MatrixXd input) {
        input = input.array().max(0);
        return input.matrix();
    } 

    Eigen::MatrixXd backward (Eigen::MatrixXd input, Eigen::MatrixXd grad) {
        for (auto &x : input.reshaped()) {
            if (x > 0) { x = 1; }
            else { x = 0; }
        }
        return input.matrix() * grad;
    }
};

#endif
