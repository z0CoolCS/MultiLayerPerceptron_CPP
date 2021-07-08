#include <iostream>
#include "Eigen/Dense"
#include "MLP.h"
#include "Utils/utils.h"
#include "Layer/ActFunc/ReLU.h"

using namespace std;
using namespace Eigen;



//void Sigmoid1(MatrixXd& X, MatrixXd& Y)
//{
	
//}


void Tanh1(MatrixXd& X, MatrixXd& Y)
{
	MLP ne;
    ne.add_layer( "Linear" , X.cols() , 15 , 0.00001);
    ne.add_layer( "tanh" );
    ne.add_layer( "linear" , 15 , 2, 0.00001); 
    
	
	ne.train(X, Y) ;		
}


void Tanh2(MatrixXd& X, MatrixXd& Y)
{
	MLP ne;
    ne.add_layer( "Linear" , X.cols() , 15 , 0.00001);
    ne.add_layer( "tanh" );
    ne.add_layer( "linear" , 15 , 2, 0.00001);
    ne.add_layer( "tanh" ); 
    //ne.add_layer( "linear" , 15 , 2, 0.00001);	
	
	ne.train( X, Y) ;		
}


void Tanh3(MatrixXd& X, MatrixXd& Y)
{
	MLP ne;
    ne.add_layer( "Linear" , X.cols() , 15 , 0.00001);
    ne.add_layer( "tanh" );
    ne.add_layer( "linear" , 15 , 15, 0.00001);
    ne.add_layer( "tanh" ); 
    ne.add_layer( "linear" , 15 , 2, 0.00001);
    ne.add_layer( "tanh" );
    
    ne.train( X, Y) ;
	
}

void Sigmoid1(MatrixXd& X, MatrixXd& Y)
{
	MLP ne;
    ne.add_layer( "Linear" , X.cols() , 15 , 0.00001);
    ne.add_layer( "sigmoid" );
    ne.add_layer( "linear" , 15 , 2, 0.00001); 
    
	
	ne.train(X, Y) ;		
}


void Sigmoid2(MatrixXd& X, MatrixXd& Y)
{
	MLP ne;
    ne.add_layer( "Linear" , X.cols() , 15 , 0.00001);
    ne.add_layer( "sigmoid" );
    ne.add_layer( "linear" , 15 , 2, 0.00001);
    ne.add_layer( "sigmoid" ); 
    //ne.add_layer( "linear" , 15 , 2, 0.00001);	
	
	ne.train( X, Y) ;		
}


void Sigmoid3(MatrixXd& X, MatrixXd& Y)
{
	MLP ne;
    ne.add_layer( "Linear" , X.cols() , 15 , 0.00001);
    ne.add_layer( "sigmoid" );
    ne.add_layer( "linear" , 15 , 15, 0.00001);
    ne.add_layer( "sigmoid" ); 
    ne.add_layer( "linear" , 15 , 2, 0.00001);
    
    ne.train( X, Y) ;
	
}

void Relu1(MatrixXd& X, MatrixXd& Y)
{
	MLP ne;
    ne.add_layer( "Linear" , X.cols() , 15 , 0.00001);
    ne.add_layer( "relu" );
    ne.add_layer( "linear" , 15 , 2, 0.00001); 
    ne.add_layer( "relu" ); 
	
	ne.train(X, Y) ;		
}


void Relu2(MatrixXd& X, MatrixXd& Y)
{
	MLP ne;
    ne.add_layer( "Linear" , X.cols() , 15 , 0.00001);
    ne.add_layer( "relu" );
    ne.add_layer( "linear" , 15 , 15, 0.00001);
    ne.add_layer( "relu" ); 
    ne.add_layer( "linear" , 15 , 2, 0.00001);	
	
	ne.train( X, Y) ;		
}


void Relu3(MatrixXd& X, MatrixXd& Y)
{
	MLP ne;
    ne.add_layer( "Linear" , X.cols() , 15 , 0.00001);
    ne.add_layer( "relu" );
    ne.add_layer( "linear" , 15 , 15, 0.00001);
    ne.add_layer( "relu" ); 
    ne.add_layer( "linear" , 15 , 2, 0.00001);
    ne.add_layer( "relu" ); 
    
    ne.train( X, Y) ;
	
}


int main () {
    
    string path = "Carncer DataSet - Hoja 1.csv";
    int rowsdata = 569;
    int colsdata = 31;
	int clases = 2;

    MatrixXd data = read_csv(rowsdata, colsdata, path);
    
    
    
    
    MatrixXd x = data( all , seq(1, last)); // caracteristicas
     
	MatrixXd y = data( all , 0); // clases , columna 0
	
	// Tanh
	
	Tanh1(x, y);
	//Tanh2(x, y);
	//Tanh3(x, y);
	
	// Sigmoid
	
	//Sigmoid1(x, y);
	//Sigmoid2(x, y);
	//Sigmoid3(x, y);
	
	
	// relu
	
	//Relu1(x, y);
	//Relu2(x, y);
	//Relu3(x, y);
    return 0;
}
