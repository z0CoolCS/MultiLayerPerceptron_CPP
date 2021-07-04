#include "../Eigen/Dense"

#ifndef _LAYER_H_
#define _LAYER_H_
class Layer {
    public: 
        virtual Eigen::MatrixXd  forward (Eigen::MatrixXd)  {Eigen::MatrixXd defaultMatrix;  return defaultMatrix;} 
        virtual Eigen::MatrixXd  backward (Eigen::MatrixXd, Eigen::MatrixXd) {Eigen::MatrixXd defaultMatrix;  return defaultMatrix;}
};

#endif
