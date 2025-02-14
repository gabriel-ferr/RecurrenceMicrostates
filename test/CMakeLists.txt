cmake_minimum_required(VERSION 3.30)
project(test)

set(CMAKE_CXX_STANDARD 26)

#       Default directory for includes, the file "RecurrenceMicrostates.h" needs to be here.
include_directories("/usr/local/include")

#       Create the executable.
add_executable(test main.cpp)

#       Find the GMP library and link it to the executable.
find_library(GMP_LIB NAMES gmp PATHS "/usr/local/lib")

if (GMP_LIB)
    target_link_libraries(test ${GMP_LIB})
else ()
    message(FATAL_ERROR "GMP library not found in /usr/local/lib")
endif ()