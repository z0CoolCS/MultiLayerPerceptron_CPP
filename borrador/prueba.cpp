 
#include <iostream> 
#include <vector> 
#include <Eigen/Dense>
#include <random>
#include <sstream> 
 #include <fstream>
 
using namespace std; 


using namespace Eigen;



MatrixXd readdata(int rows, int cols , string path)
{
	MatrixXd m(rows, cols);
	ifstream myfile (path);
	string line, word;
	int index_row = 0, index_col = 0;
	while ( getline (myfile,line) ) 
	{
        stringstream ss(line);
        index_col = 0;
        
        getline(ss, word, ','); // idomited
        
        while (getline(ss, word, ',')) 
        {
		    if (word == "M") word = "1";
		    if (word == "B") word = "2";
		    
			m(index_row, index_col) = stod(word);
			index_col++;          
        }
        index_row++;
        //cout << endl;
    }
	
	return m;
}


class Layer 
{
  public:
	
    virtual void print(){};

    virtual MatrixXd forward(MatrixXd input) {MatrixXd ans(input.rows() , input.cols() );  return ans;} 
    virtual MatrixXd backward(MatrixXd input, MatrixXd grad_output) {MatrixXd ans(input.rows() , grad_output.cols() );  return ans;}
    
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
		
		
		
		return grad_output.array() * grad_input.array();
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


MatrixXd softmax_crossentropy_with_logits( MatrixXd logits , MatrixXd reference_answers)
{
	
	

	MatrixXd logits_for_answers( logits.rows() , 1);
	MatrixXd temp =  MatrixXd::Zero( logits.rows() , 1 );
	
	for (int i = 0 ; i < logits.rows() ; i++)
	{
		
		logits_for_answers(i , 0) = logits(i , (int)reference_answers(i , 0) - 1);
		
		for (int j = 0 ; j < logits.cols() ; j++)
		{
			temp(i , 0) += exp(logits(i , j));
		}
		
		temp(i , 0) = log(temp(i , 0));
	}
	
	MatrixXd xentropy  = -logits_for_answers + temp; 
	
	return xentropy ;

}



MatrixXd grad_softmax_crossentropy_with_logits( MatrixXd logits , MatrixXd reference_answers)
{
	
	 MatrixXd ones_for_answers = MatrixXd::Zero(logits.rows(), logits.cols() );
	 
	 MatrixXd temp =  MatrixXd::Zero( logits.rows() , 1 );
	 
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




double train_batch(vector<Layer* > network , MatrixXd x , MatrixXd y)
{

	
	vector <MatrixXd> layer_activations = forward(network , x); // outputs
	
	vector <MatrixXd>layer_inputs = {x};
	layer_inputs.insert( layer_inputs.end() , layer_activations.begin() , layer_activations.end());
	MatrixXd logits = layer_activations.back(); // salida
	
	MatrixXd loss = softmax_crossentropy_with_logits(logits, y); // ERROR
	MatrixXd loss_grad  = grad_softmax_crossentropy_with_logits(logits, y);
	

	for ( int layer_index = (int)network.size() - 1 ; layer_index >= 0 ; layer_index--)
	{
		auto layer = network[layer_index];
		
		
		loss_grad = layer -> backward(layer_inputs[layer_index],loss_grad);
		
	}	
	
	return loss.mean();
	
	
}

double accuracy(MatrixXd x_test, MatrixXd y_test , vector<Layer* > network)
{
	
	MatrixXd res = x_test;
	for (auto l : network)
	{
	
		res = l -> forward ( res );
		
		
	}
	
	
	Index index;
	int check = 0;
	for (int i = 0 ; i < res.rows(); i++ )
	{
		res.row(i).maxCoeff(&index);
		
		check += index == (y_test( i , 0 ) - 1);
	}
	
	

	return (double)check / res.rows() * 100;
}



void train(vector<Layer* > network , MatrixXd x_train , MatrixXd x_valid, MatrixXd x_test, MatrixXd y_train, MatrixXd y_valid, MatrixXd y_test)
{
	

	int batch = 6;
	int epochs = 25;
	random_device rd;
    mt19937 g(rd());
	

		
	for (int e = 0 ; e < epochs ; e++)
	{
		
		cout << "                   epoch : " << e << endl; 
		vector <int> indices(x_train.rows()) ;
		iota(indices.begin() , indices.end() , 0);
		shuffle( indices.begin(), indices.end(), g);
		
	
		
		for (int start = 0 ; start < (int)indices.size() ; start += batch  )
		{
			
			vector <int> indices_batch( indices.begin() + start ,  indices.begin() + start + batch );
			
		
			train_batch( network , x_train(indices_batch, all) , y_train(indices_batch, all) );
			
			
		}
		
		cout << accuracy( x_valid , y_valid , network )  << "      " << accuracy( x_test , y_test , network ) << endl ;
		
		
	}
	
	 
}


int main()
{
	
	
	
	
	
	


	int rowsdata = 569;
	int colsdata = 31;
	int hidden = 30;
	int clases = 2;
		
	
	MatrixXd data = readdata(rowsdata, colsdata, "data.csv");
	
	MatrixXd x = data( all , seq(1, last));
	MatrixXd y = data( all , 0);
	
	

	int train_size = 0.7 * rowsdata; 
	int valid_size = 0.1 * rowsdata;

	
	MatrixXd x_valid =  x( seq( 0 , valid_size - 1) , all );	//56
	MatrixXd x_train = x( seq( valid_size , train_size - 1) , all ); // 342
	MatrixXd x_test = x( seq( train_size , last) , all ); // 171
	
	
	
	MatrixXd y_valid =  y( seq( 0 , valid_size - 1) , all );	
	MatrixXd y_train = y( seq( valid_size , train_size - 1) , all );
	MatrixXd y_test = y( seq( train_size , last) , all );
	
	

	
	
	
	vector< Layer* > network;
	
	network.push_back(new Linear( colsdata - 1 , hidden ));
	network.push_back(new Tanh);
	network.push_back(new Linear( hidden , clases ));
	
	
	
	train(network, x_train , x_valid , x_test,  y_train , y_valid , y_test);
	
	
	for (auto l : network)
	{
		delete l;
	}
	// perdida softmax
	

	
	return 0;
} 

