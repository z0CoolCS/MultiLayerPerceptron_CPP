#include "../Eigen/Dense"

#ifndef _LAYER_H_
#define _LAYER_H_
class Layer {
    public: 
        virtual void forward (Eigen::VectorXd) = 0;
        virtual void backward (Eigen::VectorXd, Eigen::VectorXd) = 0;
};

#endif
