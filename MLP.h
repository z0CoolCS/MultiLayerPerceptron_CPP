#include <iostream>
#include <vector>
#include <string>
#include <functional>
//#include <Random>
#include <fstream>
#include "Eigen/Dense"
#include <random>   
#include "Layer/ActFunc/Tanh.h"
#include "Layer/ActFunc/Sigmoid.h"
#include "Layer/ActFunc/ReLU.h"
#include "Layer/Linear.h"
#include "Layer/Layer.h"
#include "LossFunc/Softmax.h"

using namespace std;	
using namespace Eigen;
class MLP {
    std::vector<Layer*> network;

    std::vector <Eigen::MatrixXd> forward (Eigen::MatrixXd x)
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
		    vector <MatrixXd> layer_activations = forward(x); // outputs
		
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
    
    
    ~MLP()
	{
		for (auto l : network)
		{
			delete l;
		}
	}
    
    void add_layer (std::string layer , int input = 0  , int output = 0 , double learning_rate = 0.1 ) {
        transform(layer.begin(), layer.end(), layer.begin(), ::toupper);
        //Sigmoid s;
        if (layer == "SIGMOID") 
        {
			      network.emplace_back(new Sigmoid);
        } 
        else if (layer == "TANH") 
        {
			      network.emplace_back(new Tanh);
        } 
        else if (layer == "RELU") 
        {
			      network.emplace_back(new ReLU);
        }
        else if (layer == "LINEAR") 
        {
			      network.emplace_back(new Linear(input , output , learning_rate));
        }
       
    }
    
    

	  double accuracy(MatrixXd x_test, MatrixXd y_test )
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
			      check += (int)index == y_test( i , 0 );
		    }
		
		    return (double)check / res.rows() * 100;
	}
	
	
	  double get_loss(MatrixXd x_test, MatrixXd y_test )
	  {	
			MatrixXd res = x_test;
		    for (auto l : network)
		    {
			      res = l -> forward ( res );
		    }
		    
		    return softmax_crossentropy_with_logits( res , y_test).mean() ;
			
	  }
	  
	  
	  
	  
	  
	  void train(MatrixXd x_train , MatrixXd x_valid, MatrixXd x_test, MatrixXd y_train, MatrixXd y_valid, MatrixXd y_test)
	  {
			cout << "entro" << endl;
		    int batch = 64;
		    int epochs = 2000;
		    random_device rd;
		    mt19937 g(rd());
		
			ofstream out("graficas/grafica.txt");
			//out.open ;
			
			int size = x_train.rows(); 
		    for (int e = 0 ; e < epochs ; e++)
		    {
			
			       
			      vector <int> indices(x_train.rows()) ;
			      iota(indices.begin() , indices.end() , 0);
			      shuffle( indices.begin(), indices.end(), g);
					
			      for (int start = 0 ; start < (int)indices.size() ; start += batch  )
			      {
				
				        vector <int> indices_batch( indices.begin() + start ,  indices.begin() + min( start + batch , size ) );
				        train_batch( x_train(indices_batch, all) , y_train(indices_batch, all) );				
			      }
				 
				  double loss = get_loss(x_test, y_test);
				  
				  cout << "epoch  : " << e  << "       loss :  " << loss << "   "  <<   "acurraccy  : "<< accuracy(x_test, y_test) << endl; 
				  out << e + 1 << "," << loss  << "\n";
			      
		    }
		    
		    double ac = accuracy(x_test, y_test);
		    cout << "Final Acurracy :" << ac  << endl;
		    out << ac << "\n";
		    out.close();
	  }
		
		
	 void train(MatrixXd x, MatrixXd y)
	 {	
		 
		int rowsdata = x.rows(); 
		//cout << x.rows() << endl;
		
		
		// train test split
		random_device rd;
		mt19937 g(rd());
		vector<int> indices(rowsdata) ; 
		iota(indices.begin() , indices.end() , 0);
		shuffle( indices.begin(), indices.end(), g);
		
		
		  
        x = x( indices , all);
		y = y (indices , all);
	
		
		int train_size = 0.7 * rowsdata; 
		//int valid_size = 0.1 * rowsdata;

		

			
		
		//MatrixXd x_valid =  x( seq( 0 , valid_size - 1) , all );	//56
		//MatrixXd x_train = x( seq( valid_size , train_size - 1) , all ); // 342
		
		MatrixXd x_train = x( seq(   0 , train_size - 1) , all );
		MatrixXd x_test = x( seq( train_size , last) , all ); // 171
		
		
		
		//MatrixXd y_valid =  y( seq( 0 , valid_size - 1) , all );	
		//MatrixXd y_train = y( seq( valid_size , train_size - 1) , all );
		MatrixXd y_train = y( seq( 0 , train_size - 1) , all );
		MatrixXd y_test = y( seq( train_size , last) , all );
		
		
		return train( x_train , x_test , x_test,  y_train , y_test , y_test);
		//train(x_train , x_valid , x_test,  y_train , y_valid , y_test);
		
	
	 }
};
