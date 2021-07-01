#include "../Eigen/Dense"

#ifndef _LAYER_H_
#define _LAYER_H_
class Layer {
    public: 
        //Layer() { }
        virtual void forward (Eigen::VectorXd) = 0;
        virtual void backward (Eigen::VectorXd, Eigen::VectorXd) = 0;
        //void forward (Eigen::VectorXd);
        //void backward (Eigen::VectorXd, Eigen::VectorXd);
};

#endif
