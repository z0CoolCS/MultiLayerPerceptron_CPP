#include <iostream>
#include "Eigen/Dense"
//#include "../Layer.h"

#ifndef _TAN_H_
#define _TAN_H_
class Tanh: public Layer
{
	public:
		
		void print()
		{
		  cout << "This is Tanh" << endl;
		}
		
		
		
	
	MatrixXd forward(MatrixXd input) 
	{
		MatrixXd ans(input.rows() , input.cols() );
		for (int i = 0 ; i < input.rows() ; i++)
		{
			for (int j = 0 ; j < input.cols() ; j++)
			{
				ans(i , j) = tanh(input(i , j));
			}
			
		}
			
		return ans;
	}
		
		
		
		
	MatrixXd backward(MatrixXd input, MatrixXd grad_output) 
	{
		
		
		
		MatrixXd grad_input(input.rows() , input.cols() );
		for (int i = 0 ; i < input.rows() ; i++)
		{
			for (int j = 0 ; j < input.cols() ; j++)
			{
				double x = tanh(input(i , j));
				grad_input(i , j) = 1 - x*x;
				
			}
			
		
		}
		
		
		
		return grad_output.array() * grad_input.array();
	}
	
	
	
};


#endif
