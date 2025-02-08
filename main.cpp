#include <iostream>

#include "src/microstates.h"
#include "src/settings.h"
#include <unsupported/Eigen/CXX11/Tensor>

int main() {
    auto settings = new Settings({2, 2});
    Eigen::Tensor<double, 2> data(2, 10);
    for (int i = 0; i < data.dimension(0); i++) {
        for (int j = 0; j < data.dimension(1); j++) {
            data(i, j) = i * 10 + j;
        }
    }

    microstates<2>(data, data, 0.2, *settings);

    return 0;
}