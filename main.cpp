#include <iostream>
#include "Eigen/Dense"
#include "MLP.h"
//#include "Layer/ActFunc/Sigmoid.h"

using namespace std;
using namespace Eigen;

int main () {

    //MatrixXd x = MatrixXd::Random(2, 2);
    //cout << x << endl;
    //MLP a;
    //Sigmoid s;
    //VectorXd v = VectorXd::Random(2);
    //s.forward(v);




	// TEST TANH
	
	
	
	
	
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

    return 0;
}
