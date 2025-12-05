#include "include/rapidcsv.h"
#include <cerrno>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <iterator>
#include <system_error>
#include <tuple>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

Eigen::MatrixXd ReLu(Eigen::MatrixXd z){
    Eigen::MatrixXd a = z.array().max(0.0);
    
    return a;
}

Eigen::MatrixXd softmax(Eigen::MatrixXd z2){

    Eigen::RowVectorXd colmax = z2.colwise().maxCoeff();

    Eigen::MatrixXd shifted = z2.rowwise() - colmax;

    Eigen::MatrixXd exps = shifted.array().exp();
    Eigen::RowVectorXd denom = exps.colwise().sum();
    Eigen::MatrixXd pro = exps.array().rowwise() / denom.array();

    return pro;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> forward_prop(Eigen::MatrixXd w1, Eigen::VectorXd b1, Eigen::MatrixXd w2, Eigen::VectorXd b2, Eigen::MatrixXd X){
    
    Eigen::MatrixXd z1 = w1 * X;
    z1.colwise() += b1;

    Eigen::MatrixXd a1 = ReLu(z1);

    Eigen::MatrixXd z2 = w2 * a1;
    z2.colwise() += b2;

    Eigen::MatrixXd a2 = softmax(z2);

    return {z1, a1, z2, a2};
}

Eigen::MatrixXd one_hot(std::vector<float>Y){
    int m = static_cast<int>(Y.size());
    
    Eigen::MatrixXd y(10, m);
    y.setZero();

    for(int i = 0; i < m; i++){
        int c = static_cast<int>(Y[i]);
        y(c, i) = 1.0;
    }
    
    return y;
}

//Eigen::MatrixXd backpropagation_W(Eigen::MatrixXd z1, Eigen::MatrixXd a1, Eigen::MatrixXd z2, Eigen::MatrixXd a2, Eigen::MatrixXd w1, Eigen::MatrixXd w2, Eigen::MatrixXd X, std::vector<float> Y){
//    Eigen::MatrixXd onehotY = one_hot(Y);
//    int m = Y.size();
//
//    Eigen::MatrixXd dz2 = a2 - onehotY;
//    Eigen::MatrixXd dw2 = ((dz2 * a1.transpose()) / m); 
//
//    //Eigen::MatrixXd dz1 = 
//    //
//    
//    return 0;
//}

// remember to shuffle data before assigning.
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
    
    w1.setRandom();
    w2.setRandom();
    b1.setRandom();
    b2.setRandom();
    

    Eigen::MatrixXd z1, a1, z2, a2;
    std::tie(z1, a1, z2, a2) = forward_prop(w1, b1, w2, b2, X);

    std::cout << z1(2,1)<< std::endl << a1(2, 1) << std::endl << z2(2,1) << std::endl << a2(2, 1);
    
    return 0;

}


