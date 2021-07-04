#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include "Eigen/Dense"
#include "Layer/ActFunc/Sigmoid.h"
#include "Layer/Layer.h"

class MLP {
    std::vector<Layer*> network;

    std::vector <Eigen::MatrixXd> forward (Eigen::MatrixXd input)
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


    double train_batch (MatrixXd x , MatrixXd y)
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

    public:
    MLP () {
        std::cout<<1<<std::endl;
    }
    void add_layer (std::string layer, int input) {
        transform(layer.begin(), layer.end(), layer.begin(), ::toupper);
        Sigmoid s;
        if (layer == "SIGMOID") {
        } else if (layer == "TANGH") {
        } else if (layer == "RELU") {
        }
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

	  void train(vector<Layer* > network , MatrixXd x_train , MatrixXd x_valid, \
        MatrixXd x_test, MatrixXd y_train, MatrixXd y_valid, MatrixXd y_test)
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


};
