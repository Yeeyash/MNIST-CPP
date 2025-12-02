#include "include/rapidcsv.h"
#include <cmath>
#include <iostream>
// #include <Eigen/Dense>

int main(){
    
    rapidcsv::Document df("data/train.csv");
    auto col = df.GetColumn<float>("label");
    
    std::cout << col.size();

    return 0;

}


