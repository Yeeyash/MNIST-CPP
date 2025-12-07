# C++ Feedforward Nerual Network (MLP)
A lightweight, from-scratch implementation of a multilayer perceptron (MLP) n C++ using Eigen for linear algebra and rapidcsv for MNIST dataset loading. Supports forward propagation, backpropagation, ReLU activation, stable softmax, and full-batch gradient descent on handwritten digit recognition. Achieves reasonable accuracy on MNIST with a compact 784-10-10 architecture.

### Features:
- Vectorized matrix operations via Eigen (no nested loops).
- Numerically stable softmax and ReLU derivative.
- Train/test split with shuffling for MNIST (42k samples).
- Parameter saving/loading as text files for inspection.
- Reference-based API for zero-copy performance in training loop.

### Architecture
```bash
Input (784) → Hidden (10, ReLU) → Output (10, Softmax)
```
- Full-batch gradient descent (α = 0.05 - 0.2, 50 0 - 100 iterations).
- Cross-entropy loss with one-hot labels.
- Shapes:
```bash
X (Input dataset): m X 784; (m = 42000, each row : sample class, column : pixel values 0 - 783),
W<sub>1</sub>: 10 X 784,
b1<sub>1</sub?: 10,
W<sub>2</sub>: 10 X 10,
b<sub>2</sub>: 10.
```
Additional parameters:
```bash
Z<sub>1</sub>,
A<sub>1</sub>,
Z<sub>2</sub>,
A<sub>2</sub>,
dZ<sub>1</sub>,
dW<sub>1</sub>,
dB<sub>1</sub>,
dZ<sub>2</sub>,
dW<sub>2</sub>,
dB<sub>2</sub>
```

### Build Instructions
Requires Eigen3/Eigen (header-only) and rapidcsv (data manipulation).
```Windows
# Install vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Eigen3
.\vcpkg install eigen3:x64-windows

# rapidcsv (header-only, manual)
git clone https://github.com/d99kris/rapidcsv.git
# Copy rapidcsv/src/rapidcsv to your project include/

```

```macOS
# Eigen3
brew install eigen

# rapidcsv (header-only)
git clone https://github.com/d99kris/rapidcsv.git
sudo cp -r rapidcsv/src/rapidcsv /usr/local/include/
```

```Ubuntu/Debian
# Eigen3 via apt (includes headers)
sudo apt update
sudo apt install libeigen3-dev

# rapidcsv (header-only, download directly)
wget https://github.com/d99kris/rapidcsv/archive/refs/heads/master.zip
unzip master.zip
sudo cp -r rapidcsv-master/src/rapidcsv /usr/local/include/
```
#### Expected runtime: ~7 - 12 seconds for 100 iterations on CPU.

### Load dataset:
```cpp
rapidcsv::Document df("mnist.csv", rapidcsv::LabelProps::SOFT_FAIL);
Eigen::MatrixXd X(784, n);  // Pixel columns
Eigen::VectorXi y(n);       // Labels
```
### Save parameters:
```cpp
#include <fstream>  // For file I/O

void saveMatrix(const Eigen::MatrixXd& M, const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) return;  // Error handling
    
    out << M.rows() << " " << M.cols() << "\n";
    out << M << "\n";  // Eigen's built-in formatted output
    out.close();
}
saveMatrix(w1, "w1.txt");  // Portable text format
```
### Performance Tips
| Issue | Solution | Expected Gain |
|-------|----------|---------------|
| Slow training | Use references, no copies | 2-5x faster  |
| Low accuracy | Xavier init, higher LR | 80%+ test accuracy |
| NaN/Inf | Stable softmax (colmax shift) | No overflow |
| Shape errors | rowwise().sum() for biases | Matches 10×1 |
