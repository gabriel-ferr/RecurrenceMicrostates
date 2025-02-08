//
//      Recurrence Microstates - C++
//      Created by Gabriel Ferreira on 08/02/25.
//
//          Federal University of Paraná - 2025.
//      Julia Version: https://github.com/gabriel-ferr/RecurrenceMicrostates.jl
//      C++ Version: https://github.com/gabriel-ferr/RecurrenceMicrostates
#include "settings.h"
//  ----------------------------------  BODY ----------------------------------------
#include <vector>
#include <numeric>
#include <iostream>

Settings::Settings(const std::vector<int> &structure, bool force_vector, bool force_dictionaries) {
    if (structure.size() < 2)
        throw std::invalid_argument("The microstate structure required at least two dimensions.");

    //      Get the hypervolume of our space.
    hypervolume = std::accumulate(structure.begin(), structure.end(), 1, std::multiplies<int>());
    //      How the C++ lib can only can handle 64-bit data...
    if (hypervolume > 64)
        throw std::invalid_argument("The hypervolume is larger than 64. If you need to use a greater hypervolume, please try the Julia version.");

    //      Copy the given settings to our structure.
    this->structure = structure;
    this->use_dictionaries = force_dictionaries;
    this->use_dictionaries = (hypervolume > 28) ? true : use_dictionaries;
    this->use_dictionaries = force_vector ? false : use_dictionaries;
    //
    //      Compute the power vector.
    for (auto i = 0; i < hypervolume; i++)
        vect.push_back(pow(2, i));

    //      ---- DEV
    if (use_dictionaries) {
        use_dictionaries = false;
        std::cout << "The use of dictionaries is not built yet, so we change the settings to use vectors." << std::endl;
    }
}

int Settings::Size() const {
    return structure.size();
}

int Settings::Structure(const int dim) const {
    return structure[dim];
}