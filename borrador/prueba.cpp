 
#include <iostream> 
#include <vector> 
#include <Eigen/Dense>
 
 
using namespace std; 


using namespace Eigen;



class Layer 
{
  public:
	
    virtual void print() { }
    
    virtual MatrixXd forward(MatrixXd input) {MatrixXd ans(input.rows() , input.cols() );  return ans;} 
    virtual MatrixXd backward(MatrixXd input, MatrixXd grad_output) {MatrixXd ans(input.rows() , grad_output.cols() );  return ans;}
    //virtual MatrixXd backward(MatrixXd input , MatrixXd grad_output) { }
    
};


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
		
		// dE/di5 = dout/di5 * De/dout  
		return grad_output*grad_input.transpose();
	}
	
	
	
};




class Linear: public Layer
{
	public:
	
		
		MatrixXd weights;
		MatrixXd biases;
		double Learning_rate;
	
		void print()
		{
		  cout << "This is Linear" << endl;
		}
		
		
		Linear(int input_units, int output_units, double learning_rate = 0.1)
		{
			weights = 1000 * MatrixXd::Random (input_units, output_units);
			Learning_rate = learning_rate;
			biases  =  MatrixXd::Zero(1,output_units);
			
		}
		
		MatrixXd forward(MatrixXd input) 
		{
			return input * weights + biases;
		}
		
		
		MatrixXd backward(MatrixXd input, MatrixXd grad_output) 
		{
			MatrixXd grad_input =  grad_output * weights.transpose();
			
			MatrixXd grad_weights = input.transpose() * grad_output ;
			
			MatrixXd grad_biases = MatrixXd::Zero(1, grad_output.cols() );
			
			for (int i = 0 ; i < grad_output.rows() ; i++)
			{
				for (int j = 0 ; j < grad_output.rows() ; j++)
				{
					grad_biases(1, j) += grad_output(i , j);
				}
				
			}
			// MatrixXd grad_biases = grad_output.mean(axis=0)*input.shape[0];
			
			
			weights =  weights - Learning_rate * grad_weights;
			biases = biases -  learning_rate * grad_biases
			
			
			return grad_input;
		}
		
		
		
};


MatrixXd softmax_crossentropy_with_logits( MatrixXd logits , MatrixXd reference_answers)
{
	
	
	
	
	MatrixXd logits_for_answers( logits.rows() , 1);
	MatrixXd temp =  MatrixXd::Zero( logits.rows() , 1 );
	
	for (int i = 0 ; i < logits.rows() ; i++)
	{
		logits_for_answers(i , 1) = logits(i , (int)reference_answers(i , 1));
		
		for (int j = 0 ; j < logits.cols() ; j++)
		{
			temp(i , 1) += exp(logits(i , j));
		}
		
		temp(i , 1) = log(temp(i , 1));
	}
	
	MatrixXd xentropy  = -logits_for_answers + temp; 
	
	return xentropy ;
	
}



MatrixXd grad_softmax_crossentropy_with_logits( MatrixXd logits , MatrixXd reference_answers)
{
	
	 MatrixXd ones_for_answers = MatrixXd::Zero(logits.rows(), logits.cols() );
	 //MatrixXd softmax = MatrixXd::Zero(logits.rows(), logits.cols() );
	 
	 MatrixXd temp =  MatrixXd::Zero( logits.rows() , 1 );
	 
	 for (int i = 0 ; i < logits.rows() ; i++)
	{
		ones_for_answers(i ,  (int)reference_answers(i , 1) ) = 1;
		
		for (int j = 0 ; j < logits.cols() ; j++)
		{
			logits(i , j ) = exp(logits(i , j));
			temp(i , 1) += logits(i , j);
			
		}
		
		logits.row(i) /= temp(i , 1);
	}
	
	
	return (logits + ones_for_answers) / logits.rows() ;
}


vector <MatrixXd> forward ( vector<Layer *> network , MatrixXd x)
{
	vector <MatrixXd> activations ;
	
	MatrixXd input = x;
	
	for (auto l : network)
	{
		
		activations.emplace_back( l -> forward ( input ) );
		
		input = activations.back();
		
	}
	
	
	return activations;
}




double train(vector<Layer* > network , MatrixXd x , MatrixXd y)
{
	
	vector <MatrixXd> layer_activations = forward(network , x);
	vector <MatrixXd>layer_inputs = {x};
	layer_inputs.insert( layer_inputs.end() , layer_activations.begin() , layer_activations.end());
	MatrixXd logits = layer_activations.back();
	
	
	MatrixXd loss = softmax_crossentropy_with_logits(logits, y);
	MatrixXd loss_grad  = grad_softmax_crossentropy_with_logits(logits, y);
	
	
	for ( int layer_index = (int)network.size() - 1 ; layer_index >= 0 ; layer_index--)
	{
		auto layer = network[layer_index];
		
		loss_grad = layer -> backward(layer_inputs[layer_index],loss_grad);
		
	}	
	
	return loss.mean();
	
	
}



// falta leer datos
// falta iterar en minibatches
// falta predict
// crear clase network

#include <sigmoid>

class MLP
{
	vector <Layer *> cccc;
	
	
	void append( string layer )
	{
		if ()
		cccc.push_back(&la);
		else ()
		
		else ()
		
		else ()
		
		else ()
		
	}
	
	
}


int main()
{
	
	
	
	vector< Layer* > network;
	
	
	
	
	network.push_back(new Linear(4, 10));
	network.push_back(new Tanh);
	
	
	
	network.append( Linear(4, 10) );
	
	
	networl
	
	
	MatrixXd x_train = 100  * ( MatrixXd::Random( 4, 20) + MatrixXd::Constant( 4 , 20,1.)); // -100, 100
	MatrixXi y_train =  MatrixXi::Random( 4, 1) ;
	
	
	for (auto& u : y_train.reshaped())
	{
		u %= 10;
		u = abs(u);
		u += 1;
	}
	
	//cout << y_train ;
	//cout << endl << endl;
	//cout << x_train;


	train(network , x_train , y_train);
	
	for (auto u : network)
	{	
		u -> print();	
	}

	
	return 0;
} 

