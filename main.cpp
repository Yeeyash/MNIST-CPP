#include "include/rapidcsv.h"
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>
#include <iterator>
#include <sys/stat.h>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
using Eigen::MatrixXd;
using Eigen::VectorXd;

Eigen::MatrixXd ReLu(const Eigen::MatrixXd& z){
    Eigen::MatrixXd a = z.array().max(0.0);
    
    return a;
}

Eigen::MatrixXd softmax(const Eigen::MatrixXd& z2){

    Eigen::RowVectorXd colmax = z2.colwise().maxCoeff();

    Eigen::MatrixXd shifted = z2.rowwise() - colmax;

    Eigen::MatrixXd exps = shifted.array().exp();
    Eigen::RowVectorXd denom = exps.colwise().sum();
    Eigen::MatrixXd pro = exps.array().rowwise() / denom.array();

    return pro;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> forward_prop(const Eigen::MatrixXd& w1, const Eigen::VectorXd &b1, const Eigen::MatrixXd &w2, const Eigen::VectorXd& b2, const Eigen::MatrixXd &X){

    Eigen::MatrixXd z1 = w1 * X;
    z1.colwise() += b1;

    Eigen::MatrixXd a1 = ReLu(z1);

    Eigen::MatrixXd z2 = w2 * a1;
    z2.colwise() += b2;

    Eigen::MatrixXd a2 = softmax(z2);

    return {z1, a1, z2, a2};
}

Eigen::MatrixXd one_hot(const Eigen::VectorXi Y){
    int m = static_cast<int>(Y.size());
    
    Eigen::MatrixXd y(10, m);
    y.setZero();

    for(int i = 0; i < m; i++){
        int c = static_cast<int>(Y[i]);
        y(c, i) = 1.0;
    }
    
    return y;
}

Eigen::MatrixXd reluDiv(const Eigen::MatrixXd &z1){
    return (z1.array() > 0).cast<double>();
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd> backpropagation(const Eigen::MatrixXd &z1, const Eigen::MatrixXd &a1, const Eigen::MatrixXd &z2, const Eigen::MatrixXd &a2, const Eigen::MatrixXd &w1, const Eigen::MatrixXd &w2, const Eigen::MatrixXd& X, const Eigen::VectorXi &Y){
    Eigen::MatrixXd onehotY = one_hot(Y);
    int m = Y.size();
 
    Eigen::MatrixXd dz2 = a2 - onehotY;
    
    Eigen::MatrixXd dw2 = ((dz2 * a1.transpose()) / m); 
    
    Eigen::VectorXd db2 = dz2.rowwise().mean();

    //if results are incorrect check the * reluDiv portion as * is used for dot operation as well.
    Eigen::MatrixXd dz1 = (w2.transpose() * dz2).array() * reluDiv(z1).array();
    
    Eigen::MatrixXd dw1 = (dz1 * X.transpose()) / m;
    
    Eigen::VectorXd db1 = dz1.rowwise().mean();

    return {dw1, db1, dw2, db2};
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd> update_parameters(Eigen::MatrixXd& w1, Eigen::VectorXd& b1, Eigen::MatrixXd& w2, Eigen::VectorXd& b2, const Eigen::MatrixXd& dw1, const Eigen::VectorXd& db1, const Eigen::MatrixXd& dw2, const Eigen::MatrixXd& db2, double alpha){
    w1 = w1 - (alpha * dw1);
    b1 = b1 - (alpha * db1);
    w2 = w2 - (alpha * dw2);
    b2 = b2 - (alpha * db2);
    return {w1, b1, w2, b2};
}

Eigen::VectorXi predictions(const Eigen::MatrixXd& a2){
    int m = a2.cols();
    Eigen::VectorXi preds(m);
   
    for(int i = 0; i < m; i++){
        Eigen::Index maxIndex;
        a2.col(i).maxCoeff(&maxIndex);
        preds(i) = static_cast<int>(maxIndex); 
    }

    return preds;
}
double accuracy(const Eigen::VectorXi& preds, const Eigen::VectorXi& Y){
    assert(preds.size() == Y.size());
    double correct = (preds.array() == Y.array()).cast<double>().sum();

    return correct / static_cast<double>(Y.size());
}

std::tuple<Eigen::MatrixXd , Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd> gradient_descent(Eigen::MatrixXd& w1, Eigen::MatrixXd& w2, Eigen::VectorXd& b1, Eigen::VectorXd& b2, Eigen::MatrixXd& X, Eigen::VectorXi& Y, double alpha, int iterations){
    
    Eigen::MatrixXd z1, a1, z2, a2, dw1, dw2, dz1, dz2;
    Eigen::VectorXd db1(10), db2(10);
    for(int i = 0; i < iterations; i++){
        
        std::tie(z1, a1, z2, a2) = forward_prop(w1, b1, w2, b2, X);
        std::tie(dw1, db1, dw2, db2) = backpropagation(z1, a1, z2, a2, w1, w2, X, Y);
        std::tie(w1, b1, w2, b2) = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha);

        if((i % 5) == 0){
            std::cout << "Iteration: " << i << std::endl;
            Eigen::VectorXi preds = predictions(a2);
            std::cout << "Accuracy: " << accuracy(preds, Y) << std::endl;
        }
    }
    
    return {w1, b1, w2, b2};
}

int main(){
    
    rapidcsv::Document df("data/train.csv");
    int n = df.GetRowCount();
    Eigen::VectorXi y(n);
    for(int i = 0; i < n; i++){
        y(i) = df.GetCell<int>("label", i);
    }

    std::vector<std::vector<double>> rows;

    for(int i = 0; i < n; i++){
        rows.push_back(df.GetRow<double>(i));
    }

    Eigen::MatrixXd w1(10, 784);
    Eigen::VectorXd b1(10);
    Eigen::MatrixXd w2(10, 10);
    Eigen::VectorXd b2(10);
    Eigen::MatrixXd X(784, n);

    for(int i = 0; i < n; i++){
         for(int j = 0; j < 784; j++){
             X(j, i) = (rows[i][j + 1] / 255.0); 
         }
    }
    
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::mt19937 rng(std::random_device{}());
    std::shuffle(idx.begin(), idx.end(), rng);

    Eigen::MatrixXd X_shuffled(784, n);
    Eigen::VectorXi Y_shuffled(n);

    for(int i = 0; i < n; i++){
        int old_i = idx[i];

        X_shuffled.col(i) = X.col(old_i);
        Y_shuffled(i) = y(old_i);
    }

    X = std::move(X_shuffled);
    y = std::move(Y_shuffled);

    Eigen::MatrixXd X_train = X.leftCols(33600);
    Eigen::VectorXi y_train = y.head(33600);

    Eigen::MatrixXd X_test = X.rightCols(8400);
    Eigen::VectorXi y_test = y.tail(8400);

    w1.setRandom();
    w1 *= std::sqrt(2.0 / 784.0);
    w2.setRandom();
    w2 *= std::sqrt(2.0 / 10.0);
    b1.setRandom();
    b2.setRandom();
    
    Eigen::MatrixXd z1, a1, z2, a2, dw1, dw2, dz1, dz2;
    Eigen::VectorXd db1, db2;
    
    std::tie(w1, b1, w2, b2) = gradient_descent(w1, w2, b1, b2, X_train, y_train, 0.1, 100);
    //std::tie(z1, a1, z2, a2) = forward_prop(w1, b1, w2, b2, X);
    //std::tie(dw1, db1, dw2, db2) = backpropagation(z1, a1, z2, a2, w1, w2, X, y);
    
    Eigen::MatrixXd z1_test = w1 * X_test;
    z1_test.colwise() += b1;
    Eigen::MatrixXd a1_test = ReLu(z1_test);

    Eigen::MatrixXd z2_test = w2 * a1_test;
    z2_test.colwise() += b2;
    Eigen::MatrixXd a2_test = softmax(z2_test);

    Eigen::VectorXi predsTests = predictions(a2_test);
    double test_acc = accuracy(predsTests, y_test);
    std::cout << "Test accuracy: " << test_acc << std::endl;

    return 0;

}


