#include "Eigen/Dense"


// includes de las funciones extras
#include <random>
#include <sstream> 
#include <fstream>
//#include "../Gradients/Softmax.h"

#ifndef _LAYER_H_
#define _LAYER_H_

using namespace std;
using namespace Eigen;
class Layer 
{
  public:
	
    virtual void print(){};

    virtual MatrixXd forward(MatrixXd input) {MatrixXd ans(input.rows() , input.cols() );  return ans;} 
    virtual MatrixXd backward(MatrixXd input, MatrixXd grad_output) {MatrixXd ans(input.rows() , grad_output.cols() );  return ans;}
    
};


#endif
