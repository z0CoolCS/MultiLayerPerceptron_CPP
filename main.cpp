#include <iostream>
#include "Eigen/Dense"
#include "MLP.h"
#include "Layer/ActFunc/Sigmoid.h"

using namespace std;
using namespace Eigen;

int main () {

    MatrixXd x = MatrixXd::Random(2, 2);
    //cout << x << endl;
    MLP a;
    Sigmoid s;
    VectorXd v = VectorXd::Random(2);
    s.forward(v);


    return 0;
}
