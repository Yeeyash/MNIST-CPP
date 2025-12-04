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

    Eigen::MatrixXd w1(10, 784);
    Eigen::VectorXd b1(10);
    Eigen::MatrixXd w2(10, 10);
    Eigen::VectorXd b2(10);
    Eigen::MatrixXd X(784, 42000);

    for(int i = 0; i < rows.size(); i++){
         for(int j = 0; j < 784; j++){
             X(j, i) = (rows[i][j + 1] / 255.0); 
         }
    }

    std::cout << "Rows[0][4]" << rows[0][4] << "X[0][4]" << X(0, 4) << std::endl;

    w1.setRandom();
    w2.setRandom();
    b1.setRandom();
    b2.setRandom();

    Eigen::MatrixXd z1 = w1 * X;
    z1.colwise() += b1;
    //std::cout << rows[0][0] << std::endl << y[0];
    
    std::cout << z1(0, 3) << std::endl;

    return 0;

}


