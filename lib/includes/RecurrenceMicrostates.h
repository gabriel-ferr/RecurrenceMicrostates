//
//          Recurrence Microstates - C++
//          Created by Gabriel Ferreira on February 2025.
//          Advisors: Thiago de Lima Prado and Sérgio Roberto Lopes
//          Federal University of Paraná - Physics Department
//
//      Julia version: https://github.com/gabriel-ferr/RecurrenceMicrostates.jl
//      C++ version: https://github.com/gabriel-ferr/RecurrenceMicrostates
//
//          -------------------------     LIB DEFINES     -------------------------
#ifndef RECURRENCE_MICROSTATES_H
#define RECURRENCE_MICROSTATES_H
//          -----------------------------------------------------------------------
#define DEFAULT_THREADS 1
#define DEFAULT_HYPERVOLUME_TO_DICTIONARIES 28
//          -----------------------------------------------------------------------
#include <tuple>
#include <vector>
#include <thread>
#include <future>
#include <random>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <functional>
//          -----------------------------------------------------------------------
namespace RecurrenceMicrostates {
    //          -------------------------        UTILS        -------------------------
    //          Standard recurrence function.
    inline bool standard_recurrence(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &params) {
        if (params.empty()) throw std::invalid_argument("[ERROR] RecurrenceMicrostates - Recurrence: the standard recurrence needs to have one parameter: the threshold.");
        double distance = 0;
        for (auto i = 0; i < x.size(); i++)
            distance += std::pow(x[i] - y[i], 2);
        return params[0] - std::sqrt(distance) >= 0;
    }

    //          -------------------------   CLASS::SETTINGS   -------------------------
    //              Settings object to configure the recurrence computation.
    //          - structure: the base form of the microstate.
    //          - threads: number of threads to compute the probabilities.
    //          - force_vector: force use of vectors to storage probabilities.
    //          - force_dictionaries: force use of dictionaries instead vectors.
    //          - force_indexsize: size of index in bits to use as dictionaries index.
    class Settings {
        std::vector<unsigned int> vect;
        std::vector<unsigned short> structure;

        unsigned int threads = 1;
        unsigned int hypervolume = 1;
        unsigned int dictionaryIndex = 0;
        bool useDictionaries = false;

    public:
        explicit Settings(const std::vector<unsigned short> &structure, const unsigned int threads = std::thread::hardware_concurrency(), const bool force_vector = false, const bool force_dictionaries = false, const int force_indexsize = 0) {
            //      Check the input before to do anything.
            if (structure.size() < 2)
                throw std::invalid_argument("[ERROR] RecurrenceMicrostates - Settings: The microstate structure required at least two dimensions.");
            if (threads < 1) {
                std::cout << "[WARING] RecurrenceMicrostates - Settings: It was not possible to determinate the number of available threads, value set to: " << DEFAULT_THREADS << std::endl;
                this->threads = DEFAULT_THREADS;
            } else
                this->threads = threads;
            //      Save the structure.
            this->structure = structure;
            //      Get the hypervolume of our microstate.
            this->hypervolume = std::accumulate(structure.begin(), structure.end(), 1, std::multiplies());
            //      How the C++ lib only can handle 64-bit data...
            if (this->hypervolume > 64)
                throw std::invalid_argument("[ERROR] RecurrenceMicrostates - Settings: The hypervolume is larger than 64. If you need to use a greater hypervolume, please try the Julia version that can handle with 128-bit data.");
            //      Check if it needs to use dictionaries.
            this->useDictionaries = force_dictionaries;
            this->useDictionaries = hypervolume > DEFAULT_HYPERVOLUME_TO_DICTIONARIES ? true : this->useDictionaries;
            this->useDictionaries = force_vector ? false : this->useDictionaries;
            //      If we will be using dictionaries, calculates the index size.
            if (this->useDictionaries) {
                if (force_indexsize > 0) this->dictionaryIndex = force_indexsize;
                else this->dictionaryIndex = (hypervolume / 8 + 1) * 8;
            }
            //      Compute the power vector.
            for (auto i = 0; i < hypervolume; i++)
                vect.push_back(static_cast<unsigned int>(pow(2, i)));

            //      ----------  DEV: I have not implemented support for dictionaries yet =/
            if (this->useDictionaries) {
                std::cout << "[WARING] RecurrenceMicrostates - Settings: Dictionary support has not been implemented yet, so the settings will use a vector in the computational process.";
                this->useDictionaries = false;
            }
        }

        [[nodiscard]] bool UseDictionaries() const { return useDictionaries; }
        [[nodiscard]] unsigned int Size() const { return structure.size(); }
        [[nodiscard]] unsigned int Threads() const { return threads; }
        [[nodiscard]] unsigned int Hypervolume() const { return  hypervolume; }
        [[nodiscard]] unsigned int Possibilities() const { return static_cast<unsigned int>(pow(2, hypervolume)); }
        [[nodiscard]] unsigned int Power(const unsigned int index) const { return vect[index]; }
        [[nodiscard]] unsigned short int Structure(const unsigned int dim) const { return structure[dim]; }
    };
    //          -------------------------    END::SETTINGS    -------------------------

    //          -------------------------    CLASS::TENSOR    -------------------------
    template <typename type, unsigned int D> class Tensor {
        std::array<unsigned int, D> dims;
        std::vector<type> content;

        unsigned int get_index(const std::array<unsigned int, D> &indexes) const {
            auto index = indexes[0];
            for (auto i = 1; i < D; i++)
                index += indexes[i] * std::accumulate(dims.begin(), dims.begin() + i, 1, std::multiplies());
            return index;
        }

    public:
        template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == D>>
        explicit Tensor(Args... ds) : dims{static_cast<unsigned int>(ds)...} {
            const unsigned int hypervolume = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies());
            content.resize(hypervolume);
        }

        //      Operator [] to access / modify elements.
        template <typename... Indexes, typename = std::enable_if_t<sizeof...(Indexes) == D>>
        type& operator[](Indexes... indexes) {
            return content[get_index({static_cast<unsigned int>(indexes)...})];
        }

        //      Operator [] to access in const context.
        template <typename... Indexes, typename = std::enable_if_t<sizeof...(Indexes) == D>>
        const type& operator[](Indexes... indexes) const {
            return content[get_index({static_cast<unsigned int>(indexes)...})];
        }

        //      Get a vector projection of our tensor "column".
        std::vector<type> vector(std::array<unsigned int, D - 1> indexes) const {
            //      Make the "index array".
            std::array<unsigned int, D> new_indexes;
            new_indexes[0] = 0;
            std::copy(indexes.begin(), indexes.end(), new_indexes.begin() + 1);

            //      Make the result vector.
            std::vector<type> result;
            for (auto i = 0; i < dims[0]; i++) {
                result.push_back(content[get_index(new_indexes)]);
                ++new_indexes[0];
            }

            return result;
        }

        //      Get the dimensions.
        std::array<unsigned int, D> dimensions() const { return dims; }
        [[nodiscard]] unsigned int dimension(const unsigned int d) const { return dims[d]; }
    };
    //          -------------------------     END::TENSOR     -------------------------

    //          ------------------------  CLASS::PROBABILITIES ------------------------
    //              Compute the probabilities of microstates from some input data
    //          using the spacial-generalization of recurrence.
    template <unsigned short d> class Probabilities {
        std::vector<std::vector<std::vector<unsigned int>>> samples_idx;

        Settings settings;
        Tensor<double, d> data_x;
        Tensor<double, d> data_y;

        std::vector<double> vect_probs_result;

        static std::vector<double> get_data(const Tensor<double, d> data, const std::vector<unsigned int> &fix_indexes, const std::vector<unsigned int> &recursive_indexes) {
            std::array<unsigned int, d-1> indexes;
            for (unsigned int i = 0; i < d - 1; i++)
                indexes[i] = fix_indexes[i] + recursive_indexes[i];
            return data.vector(indexes);
        }

        static std::tuple<std::vector<unsigned int>, unsigned long> task_compute_using_vector(const std::vector<std::vector<unsigned int>> &samples, const Settings &settings, const Tensor<double, d> &data_x, const Tensor<double, d> &data_y,
             const std::vector<double> &params, const std::function<bool(const std::vector<double>&, const std::vector<double>&, const std::vector<double>&)> &function) {

            //      Vector to return and alloc memory to the counter.
            std::vector<unsigned int> result(settings.Possibilities(), 0);
            unsigned long counter = 0;

            //      Alloc memory to the "index add" variable and the recursive index list.
            std::vector<unsigned int> indexes(settings.Size(), 0);

            //      Get the sample indexes.
            for (auto &sample : samples) {
                unsigned int add = 0;
                std::ranges::fill(indexes, 0);

                //      Fixed indexes.
                const std::vector index_x(sample.begin(), sample.begin() + settings.Size() / 2);
                const std::vector index_y(sample.begin() + settings.Size() / 2, sample.end());

                for (auto m = 0; m < settings.Hypervolume(); m++) {
                    //      Recursive indexes.
                    const std::vector recursive_x(indexes.begin(), indexes.begin() + settings.Size() / 2);
                    const std::vector recursive_y(indexes.begin() + settings.Size() / 2, indexes.end());

                    //      Get the data.
                    std::vector<double> x = get_data(data_x, index_x, recursive_x);
                    std::vector<double> y = get_data(data_y, index_y, recursive_y);

                    add += settings.Power(m) * function(x, y, params);
                    //      Increment the recursive index.
                    indexes[0]++;
                    for (auto k = 0; k < settings.Size() - 1; k++) {
                        if (indexes[k] > settings.Structure(k) - 1) {
                            indexes[k] = 0;
                            indexes[k + 1]++;
                        }
                        else {
                            break;
                        }
                    }
                }
                result[add]++;
                counter++;
            }

            return std::make_tuple(result, counter);
        }

        void compute_using_vector(const std::vector<double> &params, const std::function<bool(const std::vector<double>&, const std::vector<double>&, const std::vector<double>&)> &function) {

            //      Alloc some memory...
            unsigned long counter = 0;

            //      Create the async tasks...
            std::vector<std::future<std::tuple<std::vector<unsigned int>, unsigned long>>> tasks;
            tasks.reserve(settings.Threads());
            for (auto i = 0; i < settings.Threads(); i++)
                tasks.push_back(std::async(std::launch::async, task_compute_using_vector, samples_idx[i], settings, data_x, data_y, params, function));

            std::vector<std::tuple<std::vector<unsigned int>, unsigned long>> results;
            results.reserve(settings.Threads());
            for (auto &f : tasks)
                results.push_back(f.get());

            for (auto &r : results)
                counter += std::get<1>(r);

            //      Alloc memory...
            this->vect_probs_result.resize(settings.Possibilities());
            for (auto &r : results) {
                auto probs = std::get<0>(r);
                for (auto i = 0; i < vect_probs_result.size(); i++)
                    this->vect_probs_result[i] += probs[i] / static_cast<double>(counter);
            }
        }

    public:
        explicit Probabilities(const Settings &settings, const Tensor<double, d> &data_x, const Tensor<double, d> &data_y, const std::vector<double> &params,
                const double samplesRate = 0.2, const std::function<bool(const std::vector<double>&, const std::vector<double>&, const std::vector<double>&)> &function = standard_recurrence): settings(settings), data_x(data_x), data_y(data_y) {

            //      Get the number of dimensions that our recurrence space has.
            const int dims = d - 1;
            const int D = 2 * dims;
            //      Check the dimension.
            if (settings.Size() != D)
                throw std::invalid_argument(
                    "[ERROR] RecurrenceMicrostates - Settings: the configured microstate structure and the given data are not compatible.");

            //      Get the data dimensions.
            auto dims_x = data_x.dimensions();
            auto dims_y = data_y.dimensions();

            //      Compute the recurrence space hypervolume.
            int rp_hypervolume = std::accumulate(dims_x.begin() + 1, dims_x.end(), 1, std::multiplies());
            rp_hypervolume *= std::accumulate(dims_y.begin() + 1, dims_y.end(), 1, std::multiplies());
            //      Number of samples that we have.
            const auto number_of_samples = static_cast<unsigned int>(std::floor(samplesRate * rp_hypervolume));

            //      Get the samples.
            std::vector samples(number_of_samples, std::vector<unsigned int>(D, 0));

            std::random_device rd;
            std::mt19937 gen(rd());

            for (auto dim = 0; dim < dims; dim++) {
                const int max_x = dims_x[dim + 1] - settings.Structure(dim);
                const int max_y = dims_y[dim + 1] - settings.Structure(dim);

                std::uniform_int_distribution<unsigned int> dist_x(0, max_x);
                std::uniform_int_distribution<unsigned int> dist_y(0, max_y);

                for (auto i = 0; i < number_of_samples; i++) {
                    samples[i][dim] = dist_x(gen);
                    samples[i][dims + dim] = dist_y(gen);
                }
            }

            //      We divide the work between each available thread...
            const unsigned int int_numb = static_cast<int>(number_of_samples / settings.Threads());
            unsigned int rest_numb = number_of_samples % settings.Threads();
            samples_idx.resize(settings.Threads());

            unsigned int start_idx = 0;
            for (auto i = 0; i < settings.Threads(); i++) {
                const unsigned int numb = int_numb + (rest_numb > 0 ? 1 : 0);

                samples_idx[i].resize(numb);
                std::vector<int> samples_idx_tmp(numb);
                std::iota(samples_idx_tmp.begin(), samples_idx_tmp.end(), start_idx);

                for (auto j = 0; j < numb; j++) samples_idx[i][j] = samples[samples_idx_tmp[j]];

                start_idx += numb;
                if (rest_numb > 0) --rest_numb;
            }

            if (settings.UseDictionaries()) throw std::runtime_error("[ERROR] RecurrenceMicrostates - Settings: Dictionaries has not been implemented yet.");
            compute_using_vector(params, function);
        }

        [[nodiscard]] std::vector<double> Distribution() const { return this->vect_probs_result; }
    };
    //          -------------------------  END::PROBABILITIES -------------------------
}
#endif RECURRENCE_MICROSTATES_H
//          -------------------------      LIB END        -------------------------