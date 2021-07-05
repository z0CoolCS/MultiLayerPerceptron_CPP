#include <iostream>
#include "Eigen/Dense"
#include "MLP.h"
#include "Utils/utils.h"
#include "Layer/ActFunc/ReLU.h"

using namespace std;
using namespace Eigen;

int main () {
    
    string path = "Carncer DataSet - Hoja 1.csv";
    int rowsdata = 569;
    int colsdata = 31;
	int clases = 2;

    MatrixXd data = read_csv(rowsdata, colsdata, path);
    

    
    
    //MatrixXd data = data2(seq(1, 100), seq(0, 4));
    
    MatrixXd x = data( all , seq(1, last)); // caracteristicas
	 
	MatrixXd y = data( all , 0); // clases , columna 0
	
    //cout << x;
    
    //cout << x << endl;
    //cout << y << endl;
    
    // 0.5 0.6
    MLP a;
    a.add_layer( "Linear" , x.cols() , 15 , 0.00001);
    a.add_layer( "tanh" );
    a.add_layer( "linear" , 15 , 2, 0.00001); 

	a.train(x, y) ;
    //MatrixXd tmp(2, 3);
    //tmp << 0, 0, -1, 1,1,1;
    //cout << tmp << endl;
    //ReLU d;
    //cout<<d.backward(tmp, tmp);

    return 0;
}
