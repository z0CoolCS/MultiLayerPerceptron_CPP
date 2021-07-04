 
#include <iostream> 
#include <vector> 
#include <Eigen/Dense>
#include <random>
#include <sstream> 
 #include <fstream>
 
using namespace std; 


using namespace Eigen;



class Layer 
{
  public:
	
    virtual void print() ;
    
    virtual MatrixXd forward(MatrixXd input );
    virtual MatrixXd backward(MatrixXd input, MatrixXd grad_output) ;
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
		
		
		// DO/DI
		MatrixXd grad_input(input.rows() , input.cols() );
		for (int i = 0 ; i < input.rows() ; i++)
		{
			for (int j = 0 ; j < input.cols() ; j++)
			{
				double x = tanh(input(i , j));
				grad_input(i , j) = 1 - x*x;
				//cout << x << " ";
			}
			
			//cout << endl;
		}
		
		
		// d
		//cout << "tanh backward " << endl;
		//cout << input.rows() << " " << input.cols() << endl;
		//cout << grad_output.rows() << " " << grad_output.cols() << endl;
		
		//cout << "grad output tahn" << endl;
		//cout << grad_output << endl ;
		//cout << endl;
		
		//cout << "grad input tahn" << endl;
		//cout << grad_input << endl ;
	
		// array = matriz;
		// O(1) ;
		// Matrix
		// 
		
		// dot product Matrix * matrix operacion Algebra lineal
		// coefficiente product  Array * array;  operacio coef wise exp
		
		
	
		// o o o o 
		// 
		// 
		// 
		// []
		//return graout.array().exp();
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
			weights = MatrixXd::Random (input_units, output_units) * 0.1;
			Learning_rate = learning_rate;
			biases  =  MatrixXd::Zero(1,output_units);
			
		}
		
		MatrixXd forward(MatrixXd input) 
		{
			// 4 x 20  *  20  x 10 + 1 * 10;
			
			//   1 2 3 4 5 ... 20
			// 1
			// 2
			// 3
			// .
			// .
			// .
			// 4 
			
			
			
			// 4
			//return input * weights + biases;
			MatrixXd output = input * weights;
			
			//cout << output.rows() << " " << output.cols() << endl; 
			//cout << biases.rows() << " " << biases.cols() << endl; 
			
			
			output.rowwise() += biases.reshaped().transpose();
			
			return output;
		}
		
		
		MatrixXd backward(MatrixXd input, MatrixXd grad_output) 
		{
			//cout << ga
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
			// MatrixXd grad_biases = grad_output.mean(axis=0)*input.shape[0];
			
			//cout << weights << endl;
			
			//cout << endl;
			
			//cout << "gout" << endl;
			//cout << grad_weights << endl;
			
			//cout << endl;
			weights =  weights - Learning_rate * grad_weights;
			biases = biases -  Learning_rate * grad_biases;
			
			
			//cout << weights << endl;
		
			
			return grad_input;
		}
		
		
		
};


MatrixXd softmax_crossentropy_with_logits( MatrixXd logits , MatrixXd reference_answers)
{
	
	
	
	// 4 x 10 - 4 x 1;
	//cout << "ga" << endl;
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
	 //MatrixXd softmax = MatrixXd::Zero(logits.rows(), logits.cols() );
	 
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
		//l -> print();
		activations.emplace_back( l -> forward ( input ) );
		
		input = activations.back();
		
		//l -> print();
		
	}
	
	
	return activations;
}




double train_batch(vector<Layer* > network , MatrixXd x , MatrixXd y)
{
	
	//cout << "ga";
	vector <MatrixXd> layer_activations = forward(network , x); // outputs
	vector <MatrixXd>layer_inputs = {x};
	layer_inputs.insert( layer_inputs.end() , layer_activations.begin() , layer_activations.end());
	MatrixXd logits = layer_activations.back(); // salida
	

	 
	MatrixXd loss = softmax_crossentropy_with_logits(logits, y); // ERROR
	MatrixXd loss_grad  = grad_softmax_crossentropy_with_logits(logits, y);
	
	
	// loss_Grad = dE/ salida
	// DE / DI(tanh) =  lossgrad* Dsalida/DItanh
	//cout << "logits : " << endl;
	for ( int layer_index = (int)network.size() - 1 ; layer_index >= 0 ; layer_index--)
	{
		auto layer = network[layer_index];
		
		
		loss_grad = layer -> backward(layer_inputs[layer_index],loss_grad);
		
	}	
	//cout << "finish : " << endl;
	
	return loss.mean();
	
	//return 1.0;
}





void train(vector<Layer* > network , MatrixXd x_train , MatrixXd x_valid, MatrixXd y_train, MatrixXd y_valid)
{
	
	// cmbiar para batch != divisor de examples;
	int batch = 5;
	int epochs = 2;
	random_device rd;
    mt19937 g(rd());
	
	
	for (int e = 0 ; e < epochs ; e++)
	{
		
		//cout << "eposch : " << e << endl; 
		vector <int> indices(x_train.rows()) ;
		iota(indices.begin() , indices.end() , 0);
		shuffle( indices.begin(), indices.end(), g);
		
		for (int start = 0 ; start < (int)indices.size() ; start += batch  )
		{
			//cout << start << endl;
			vector <int> indices_batch( indices.begin() + start ,  indices.begin() + start + batch );
			
			
			double error = train_batch( network , x_train(indices_batch, all) , y_train(indices_batch, all) );
			
			cout << error << endl;
		}
		
		// validar
		
	}
	
	 
}


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


// 10 clases
// output final 2; [ 0.55, 0.66] = 1

// falta predict
// crear clase network

int main()
{
	
	
	//MatrixXd x_train = 100  * ( MatrixXd::Random( examples , caracs) + MatrixXd::Constant( examples , caracs ,1.)); // -1, 1
	//MatrixXd y_train =  100 * MatrixXd::Random( examples, 1) ;
	
	

	
	
	//for (auto& u : y_train.reshaped())
	//{
		//u = abs(u);
		//u =  (int)u % images;
		//u = floor(u);
		//u += 1;
	//}
	

	
	//MatrixXd x_train_train = x_train( seq(0, 129) , all);
	//MatrixXd x_train_valid = x_train( seq(130, 149) , all);
	
	
	//MatrixXd y_train_train = y_train( seq(0, 129) , all);
	//MatrixXd y_train_valid = y_train( seq(130, 149) , all);
	
	// este entre 1 y 10;
	
	// Valid 10%
	
	
	


	int rowsdata = 569;
	int colsdata = 31;
	int hidden = 30;
	int clases = 10;
		
	
	MatrixXd data = readdata(rowsdata, colsdata, "data.csv");
	
	MatrixXd x = data( all , seq(1, last));
	MatrixXd y = data( all , 0);
	
	
	//int train_size = 0.6 * rowsdata;
	int train_size = 0.7 * rowsdata;
	//int test_size = rowdata - train_size;
	
	MatrixXd x_train = x( seq( 0 , train_size - 1) , all );
	MatrixXd x_test = x( seq( train_size , last) , all );
	
	
	
	//MatrixXd x_train_train = x_train();
	//MatrixXd x_train_valid = 
	
	
	
	
	MatrixXd y_train = y( seq( 0 , train_size - 1) , all );
	MatrixXd y_test = y( seq( train_size , last) , all );
	
	
	vector< Layer* > network;
	
	network.push_back(new Linear( colsdata , hidden ));
	network.push_back(new Tanh);
	network.push_back(new Linear( hidden , clases ));
	network.push_back(new Tanh);
	// perdida softmax
	
	
	
	
	
	
	//train(network, x_train_train , x_train_valid , y_train_train , y_train_valid );

	
	

	
	return 0;
} 

