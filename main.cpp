#include "include/rapidcsv.h"
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <iterator>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

int ReLu(int x){
    if(x > 0) return x;
    return 0;
}

int main(){
    
    rapidcsv::Document df("data/train.csv");
    auto y = df.GetColumn<float>("label");
    int n = df.GetRowCount();

    std::vector<std::vector<double>> rows;

    for(int i = 0; i < n; i++){
        rows.push_back(df.GetRow<double>(i));
    }

    std::cout << rows[0][0] << std::endl;

//    Eigen::MatrixXd w1(10, 784);
//    Eigen::MatrixXd b1(10, 1);
//    Eigen::MatrixXd w2(10, 10);
//    Eigen::MatrixXd b2(10, 1);
    
//    w1.setRandom();
//    w2.setRandom();
//    b1.setRandom();
//    b2.setRandom();

//    Eigen::MatrixXd z1 = (w1.array() * x.array()).matrix();


    return 0;

}



