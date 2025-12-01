#include "include/rapidcsv.h"
#include <iostream>
#include <Eigen/Dense>

int main(){
    
    Eigen::MatrixXd m(2,2);
    m << 1, 2, 3, 4;
    std::cout << "Eigen works" << m << std::endl;

    rapidcsv::Document doc("data/dummy.csv", rapidcsv::LabelParams(0,0));
    std::cout << "rapidcsv header found" << std::endl;

    return 0;

}


