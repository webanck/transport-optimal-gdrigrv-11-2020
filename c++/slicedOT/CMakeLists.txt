PROJECT(OTColorTransfer)
cmake_minimum_required(VERSION 3.1)

set (CMAKE_CXX_STANDARD 17)

find_package(OpenMP)

set(EXAMPLES
  colorTransfer
)

foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    if(UNIX)
      target_link_libraries(${EXAMPLE} -lm)
    endif()
    if(OpenMP_CXX_FOUND)
      target_link_libraries(${EXAMPLE} OpenMP::OpenMP_CXX)
    endif()
endforeach()
