#include <iostream>
#include "Eigen/Dense"
#include "MLP.h"
#include "Utils/utils.h"

using namespace std;
using namespace Eigen;

int main () {


	cout << "ga" << endl;
    string path = "Carncer DataSet - Hoja 1.csv";
    int rowsdata = 569;
    int colsdata = 31;
	int clases = 2;

    MatrixXd data = read_csv(rowsdata, colsdata, path);
    
    MatrixXd x = data( all , seq(1, last)); // caracteristicas
	 
	MatrixXd y = data( all , 0); // clases , columna 0
	
	
    
    //cout << x;
    MLP a;
    a.add_layer( "Linear" , x.cols() , 15 );
    a.add_layer( "Tanh" , 15, 2 );


	a.train(x, y) ;

    return 0;
}
