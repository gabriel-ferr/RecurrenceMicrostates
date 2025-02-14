#include <gmp.h>
#include <iostream>
#include <RecurrenceMicrostates.h>

int main() {
    std::cout << "[TESTING] Recurrence Microstates C++ Library" << std::endl;

    try {
        RecurrenceMicrostates::Tensor<double, 2> data(2, 100);
        //for (auto k = 0; k < data.dimension(2); k++) {
            for (auto j = 0; j < data.dimension(1); j++) {
                for (auto i = 0; i < data.dimension(0); i++) {
                    data[i, j] = cos(3.4 * (i - j)) + sin(3.14 * j);
                }
            }
        //}

        std::cout << "[PASSED] Tensor structures is working." << std::endl;
        const auto settings = std::make_unique<RecurrenceMicrostates::Settings>(std::vector<unsigned short>{2, 2}, 1);
        std::cout << "[PASSED] Setting structure created for vectors." << std::endl;
        const auto probabilities = std::make_unique<RecurrenceMicrostates::Probabilities<2>>(*settings, data, data, std::vector<double>{0.2});
        std::cout << "[PASSED] Probabilities computed for vectors." << std::endl;
        for (const auto distribution = probabilities->Distribution(); auto prob : distribution)
            std::cout << prob << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[FAIL]: " << e.what() << std::endl;
    }
    return 0;
}