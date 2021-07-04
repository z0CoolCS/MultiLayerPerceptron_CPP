#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "../Eigen/Dense"

Eigen::MatrixXd read_csv(int rows, int cols , std::string path)
{

	Eigen::MatrixXd m(rows, cols);
	std::ifstream myfile (path);
	std::string line, word;
	int index_row = 0, index_col = 0;

	while ( getline (myfile,line) ) 
	{
        std::stringstream ss(line);
        index_col = 0;
        
        getline(ss, word, ','); // id omited
        
        while (getline(ss, word, ',')) 
        {
		        if (word == "M") word = "0";
		        else if (word == "B") word = "1";
		    
			      m(index_row, index_col) = stod(word);
			      index_col++;          
        }

        index_row++;
        if (index_row == rows) { break; }
    }
	
	return m;

}
