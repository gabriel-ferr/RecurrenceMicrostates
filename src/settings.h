//
//      Recurrence Microstates - C++
//      Created by Gabriel Ferreira on 08/02/25.
//
//          Federal University of Paraná - 2025.
//      Julia Version: https://github.com/gabriel-ferr/RecurrenceMicrostates.jl
//      C++ Version: https://github.com/gabriel-ferr/RecurrenceMicrostates
#ifndef SETTINGS_H
#define SETTINGS_H

#include <vector>
//      Settings object to configure the recurrence computation.
class Settings {
private:
    std::vector<int> vect;
    std::vector<int> structure;

    int hypervolume = 1;
    bool use_dictionaries = false;

public:
    Settings(const std::vector<int>& structure, bool force_vector = false, bool force_dictionaries = false);
    int Size() const;
    int Structure(const int dim) const;
};

#endif //SETTINGS_H
