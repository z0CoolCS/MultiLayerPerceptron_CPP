#include <iostream>
#include "Eigen/Dense"
#include "MLP.h"
#include "Utils/utils.h"

using namespace std;
using namespace Eigen;

int main () {

    string path = "data_proy4.csv";
    int rowsdata = 568;
    int colsdata = 31;

    MatrixXd x = read_csv(rowsdata, colsdata, path);
    //MLP a;
    //a.add_layer("Sigmoid");


    return 0;
}
