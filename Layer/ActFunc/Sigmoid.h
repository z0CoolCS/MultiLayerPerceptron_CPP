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
        //std::cout<<input.exp()<<std::endl;
    } 

    Eigen::MatrixXd backward (Eigen::MatrixXd input, Eigen::MatrixXd grad) {
    }
};

#endif
