//
//      Recurrence Microstates - C++
//      Created by Gabriel Ferreira on 08/02/25.
//
//          Federal University of Paraná - 2025.
//      Julia Version: https://github.com/gabriel-ferr/RecurrenceMicrostates.jl
//      C++ Version: https://github.com/gabriel-ferr/RecurrenceMicrostates
#ifndef MICROSTATES_H
#define MICROSTATES_H

#include "settings.h"

#include <vector>
#include <random>
#include <thread>
#include <numeric>
#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>

//      Standard recurrence function.
bool recurrence(const std::vector<double> &x, const std::vector<double> &y, double threshold) {
    std::cout << "recurrence called" << std::endl;
}

template <int d>
//      Script that compute the microstates probabilities.
std::vector<double> microstates(const Eigen::Tensor<double, d> &data_x, const Eigen::Tensor<double, d> &data_y, double threshold, Settings settings,
        double samplePercent = 0.2, std::function<bool(const std::vector<double>&, const std::vector<double>&, double)> func = recurrence, const int threads = std::thread::hardware_concurrency()) {

    const int dims = d - 1;
    const int D = 2 * dims;

    if (settings.Size() != D)
        throw std::runtime_error("The configured Power Vector and the given data are not compatible.");
    if (threads <= 0)
        throw std::runtime_error("The number of threads must be greater than 0.");

    auto dims_x = data_x.dimensions();
    auto dims_y = data_y.dimensions();

    int rp_hypervolume = std::accumulate(dims_x.begin() + 1, dims_x.end(), 1, std::multiplies<int>());
    rp_hypervolume *= std::accumulate(dims_y.begin() + 1, dims_y.end(), 1, std::multiplies<int>());

    int samples_n = static_cast<int>(std::floor(samplePercent * rp_hypervolume));
    std::vector<std::vector<int>> samples(samples_n, std::vector<int>(D, 0));

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int dim = 0; dim < dims; dim++) {
        const int max_x = dims_x[dim + 1] - settings.Structure(dim);
        const int max_y = dims_y[dim + 1] - settings.Structure(dim);

        std::uniform_int_distribution<int> dist_x(0, max_x);
        std::uniform_int_distribution<int> dist_y(0, max_y);

        for (int i = 0; i < samples_n; i++) {
            samples[i][dim] = dist_x(gen);
            samples[i][dims + dim] = dist_y(gen);
        }
    }

    const int int_numb = static_cast<int>(std::floor(samples_n / threads));
    const int rest_samples = samples_n - threads * int_numb;

    std::cout << int_numb << ", " << rest_samples << std::endl;
}

#endif //MICROSTATES_H
