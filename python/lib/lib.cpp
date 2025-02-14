//
//          Recurrence Microstates - C++
//          Created by Gabriel Ferreira on February 2025.
//          Advisors: Thiago de Lima Prado and Sérgio Roberto Lopes
//          Federal University of Paraná - Physics Department
//
//      Julia version: https://github.com/gabriel-ferr/RecurrenceMicrostates.jl
//      C++ version: https://github.com/gabriel-ferr/RecurrenceMicrostates
//
//          This is the python integration interface written in C++ using PyBind11.
//          -------------------------     LIB INCLUDES     -------------------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
//          -----------------------------------------------------------------------
#include <RecurrenceMicrostates.h>
//          -----------------------------------------------------------------------
#include <vector>
//          -----------------------------------------------------------------------
//                  Convert a NumPy array to a 2D tensor.
RecurrenceMicrostates::Tensor<double, 2> numpy_to_2d_tensor(const std::vector<unsigned int> &dimensions, const std::vector<double> &content) {

    //      Instantiate our tensor.
    RecurrenceMicrostates::Tensor<double, 2> result(dimensions[0], dimensions[1]);

    //      Get the elements...
    for (unsigned int i = 0; i < dimensions[0]; i++) {
        for (unsigned int j = 0; j < dimensions[1]; j++) {
            const unsigned int index = i * dimensions[1] + j;
            result[i, j] = content[index];
        }
    }

    return result;
}
//          -----------------------------------------------------------------------
//                  Convert a NumPy array to a 2D tensor.
RecurrenceMicrostates::Tensor<double, 3> numpy_to_3d_tensor(const std::vector<unsigned int> &dimensions, const std::vector<double> &content) {

    //      Instantiate our tensor.
    RecurrenceMicrostates::Tensor<double, 3> result(dimensions[1], dimensions[2], dimensions[0]);

    //      Get the elements...
    for (unsigned int i = 0; i < dimensions[1]; i++) {
        for (unsigned int j = 0; j < dimensions[2]; j++) {
            for (unsigned int k = 0; k < dimensions[0]; k++) {
                const unsigned int index = k * (dimensions[1] * dimensions[2]) + i * dimensions[2] + j;
                result[i, j, k] = content[index];
            }
        }
    }

    return result;
}
//          -----------------------------------------------------------------------
//                  Define the settings of our recurrence library.
pybind11::capsule settings(const pybind11::tuple &structure, const unsigned int threads = DEFAULT_THREADS, const bool force_vector = false, const bool force_dictionary = false, const size_t force_indexsize = 0) {
    const auto conf = new RecurrenceMicrostates::Settings(structure.cast<std::vector<unsigned short>>(), threads, force_vector, force_dictionary, force_indexsize);
    return {conf, "settings_ptr", [](void* ptr) {
        delete static_cast<RecurrenceMicrostates::Settings*>(ptr);
    }};
}
//          -----------------------------------------------------------------------
//                  Compute the microstates probabilities of some data.
pybind11::array_t<double> probabilities(const pybind11::capsule &config, const pybind11::array_t<double> &data_x, const pybind11::array_t<double> &data_y, const double threshold, const double sample_rate = 0.2) {
    //      Load the settings from the pointer.
    auto settings = static_cast<RecurrenceMicrostates::Settings *>(config.get_pointer());

    //      Get a buffer info to safe access.
    const auto info_x = data_x.request();
    const auto info_y = data_y.request();

    //      Get the number of dimensions.
    const auto dims_x = info_x.ndim;
    const auto dims_y = info_y.ndim;

    //      If the number of dimensions is different, we have an error.
    if (dims_x != dims_y) throw std::invalid_argument("[ERROR] RecurrenceMicrostates - Data: The number of dimensions of 'data_x' and 'data_y' is different.");

    //      Define our result vector.
    std::vector<double> probs;

    //      Get the data shape and content.
    const std::vector<unsigned int> shape_x(info_x.shape.begin(), info_x.shape.end());
    const std::vector content_x(static_cast<double*>(info_x.ptr), static_cast<double*>(info_x.ptr) + info_x.size);
    const std::vector<unsigned int> shape_y(info_y.shape.begin(), info_y.shape.end());
    const std::vector content_y(static_cast<double*>(info_y.ptr), static_cast<double*>(info_y.ptr) + info_y.size);

    //      Compute the probs for each case...
    if (dims_x == 2) {
        const auto x = numpy_to_2d_tensor(shape_x, content_x);
        const auto y = numpy_to_2d_tensor(shape_y, content_y);

        //      Compute the probabilities...
        auto probabilities = std::make_unique<RecurrenceMicrostates::Probabilities<2>>(*settings, x, y, std::vector{threshold}, sample_rate);
        probs = probabilities->Distribution();
    } else if (dims_x == 3) {
        const auto x = numpy_to_3d_tensor(shape_x, content_x);
        const auto y = numpy_to_3d_tensor(shape_y, content_y);

        //      Compute the probabilities...
        auto probabilities = std::make_unique<RecurrenceMicrostates::Probabilities<3>>(*settings, x, y, std::vector{threshold}, sample_rate);
        probs = probabilities->Distribution();
    } else {
        throw std::runtime_error("[ERROR] RecurrenceMicrostates - Python: The Python interface just has support for 2 or 3 dimensions. If you need, try the Julia or C++ versions.");
    }

    //      Move the result probabilities to the "memory heap" for we access it from the python =3
    auto result = std::make_unique<std::vector<double>>(probs);

    //      Return the probabilities as a NumPy array.
    return pybind11::array_t<double>(
        {static_cast<pybind11::ssize_t>(result->size())},
        {sizeof(double)},
        result->data(),
        pybind11::capsule(result.release(), [](void *ptr) { delete static_cast<std::vector<double> *>(ptr); })
    );
}
//          -----------------------------------------------------------------------
//                  Overload for a normal RP instead a CRP.
pybind11::array_t<double> probabilities(const pybind11::capsule &config, const pybind11::array_t<double> &data, const double threshold, const double sample_rate = 0.2) {
    return probabilities(config, data, data, threshold, sample_rate);
}
//          -----------------------------------------------------------------------
//                  Python Module (using PYBIND11)
PYBIND11_MODULE(recurrms, m) {
    m.def("settings", &settings,
        pybind11::arg("structure"),
        pybind11::arg("threads") = DEFAULT_THREADS,
        pybind11::arg("force_vector") = false,
        pybind11::arg("force_dictionary") = false,
        pybind11::arg("force_indexsize") = 0,
        "Create a setting structure to compute the microstates of any data.");

    m.def("probabilities",
        pybind11::overload_cast<const pybind11::capsule&, const pybind11::array_t<double>&, const pybind11::array_t<double>&, const double, const double>(&probabilities),
        pybind11::arg("config"),
        pybind11::arg("data_x"),
        pybind11::arg("data_y"),
        pybind11::arg("threshold"),
        pybind11::arg("sample_rate") = 0.2,
        "Return a NumPy array with the result probabilities.");

    m.def("probabilities",
       pybind11::overload_cast<const pybind11::capsule&, const pybind11::array_t<double>&, const double, const double>(&probabilities),
       pybind11::arg("config"),
       pybind11::arg("data"),
       pybind11::arg("threshold"),
       pybind11::arg("sample_rate") = 0.2,
       "Return a NumPy array with the result probabilities.");
}
