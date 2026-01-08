cmake_minimum_required(VERSION 3.15)
project(libantipodal VERSION 0.1.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    quotient_space.cpp
    parity_index.cpp
    backends/faiss_backend.cpp
)

# Create library
add_library(antipodal SHARED ${SOURCES})

# Include directories
target_include_directories(antipodal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optional: Use real FAISS if available
find_package(faiss QUIET)
if(faiss_FOUND)
    target_link_libraries(antipodal PRIVATE faiss)
    target_compile_definitions(antipodal PRIVATE USE_FAISS)
    message(STATUS "Building with FAISS support")
else()
    message(STATUS "Building with stub FAISS implementation")
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(antipodal PRIVATE -O3 -march=native)
endif()

# Install
install(TARGETS antipodal
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    quotient_space.h
    parity_index.h
    backends/faiss_backend.h
    DESTINATION include/antipodal
)
