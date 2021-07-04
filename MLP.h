#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "Layer/ActFunc/Sigmoid.h"
#include "Layer/ActFunc/Tanh.h"
#include "Layer/ActFunc/Linear.h"
#include "Gradients/Softmax.h"
//#include "Layer/Layer.h"




using namespace std;
class MLP {
    //std::vector<*Layer>
    public:
    MLP () {
        std::cout<<1<<std::endl;
    }
    void add_layer () {
    }

};





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
