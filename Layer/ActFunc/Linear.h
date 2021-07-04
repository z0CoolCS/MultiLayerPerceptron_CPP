#include <iostream>
#include "../../Eigen/Dense"
#include "../Layer.h"

using namespace std;
#ifndef _LINEAR_H_
#define _LINEAR_H_


using namespace Eigen;
class Linear: public Layer
{
	public:
	
		
		MatrixXd weights;
		MatrixXd biases;
		double Learning_rate;
	
		
		Linear(int input_units, int output_units, double learning_rate = 0.1)
		{
			
			weights = MatrixXd::Random (input_units, output_units) ;
			Learning_rate = learning_rate;
			biases  =  MatrixXd::Zero(1,output_units);
			
		}
		
		MatrixXd forward(MatrixXd input) 
		{
		
			MatrixXd output = input * weights;
			
			
			
			output.rowwise() += biases.reshaped().transpose();
			
			return output;
		}
		
		
		MatrixXd backward(MatrixXd input, MatrixXd grad_output) 
		{
			
			MatrixXd grad_input =  grad_output * weights.transpose();
			
			// WT = DO/DI;
			
			// DE / Di = DE/ DO * DO/ DI
			
			MatrixXd grad_weights = input.transpose() * grad_output ;
			
			MatrixXd grad_biases = MatrixXd::Zero(1, grad_output.cols() ); // 1 * 10
			
			for (int i = 0 ; i < grad_output.rows() ; i++)
			{
				for (int j = 0 ; j < grad_output.cols() ; j++)
				{
					grad_biases(0, j) += grad_output(i , j);
				}
				
			}
			
			weights =  weights - Learning_rate * grad_weights;
			biases = biases -  Learning_rate * grad_biases;
			
			
			return grad_input;
		}
		
		
		
};


#endif
