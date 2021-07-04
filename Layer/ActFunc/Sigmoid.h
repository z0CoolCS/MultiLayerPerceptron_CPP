#include <iostream>
#include "Eigen/Dense"
#include "../Layer.h"

#ifndef _SIGMOIF_H_
#define _SIGMOIF_H_
class Sigmoid : public Layer {
    public:
    Sigmoid () {
        std::cout<<1<<std::endl;
    }

    void forward (Eigen::VectorXd input) {
        //std::cout<<input.exp()<<std::endl;
    } 

    void backward (Eigen::VectorXd input, Eigen::VectorXd grad) {
    }
};

#endif
