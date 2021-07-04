#include <iostream>
#include "../Eigen/Dense"

Eigen::MatrixXd softmax_crossentropy_with_logits( Eigen::MatrixXd logits , Eigen::MatrixXd reference_answers)
{
	Eigen::MatrixXd logits_for_answers( logits.rows() , 1);
	Eigen::MatrixXd temp =  MatrixXd::Zero( logits.rows() , 1 );
	
	for (int i = 0 ; i < logits.rows() ; i++)
	{
		
		logits_for_answers(i , 0) = logits(i , (int)reference_answers(i , 0) - 1);
		
		for (int j = 0 ; j < logits.cols() ; j++)
		{
			temp(i , 0) += exp(logits(i , j));
		}
		
		temp(i , 0) = log(temp(i , 0));
	}
	
	Eigen::MatrixXd xentropy  = -logits_for_answers + temp; 
	
	return xentropy ;

}

Eigen::MatrixXd grad_softmax_crossentropy_with_logits( Eigen::MatrixXd logits , Eigen::MatrixXd reference_answers)
{
	
	 Eigen::MatrixXd ones_for_answers = Eigen::MatrixXd::Zero(logits.rows(), logits.cols() );
	 
	 Eigen::MatrixXd temp =  Eigen::MatrixXd::Zero( logits.rows() , 1 );
	 
	for (int i = 0 ; i < logits.rows() ; i++)
	{
		ones_for_answers(i ,  (int)reference_answers(i , 0)  - 1 ) = 1;
		
		for (int j = 0 ; j < logits.cols() ; j++)
		{
			logits(i , j ) = exp(logits(i , j));
			temp(i , 0) += logits(i , j);
			
		}
		
		logits.row(i) /= temp(i , 0);
	}
	
	
	return (logits + ones_for_answers) / logits.rows() ;
}
