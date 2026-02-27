cmake_minimum_required(VERSION 3.16)
project(csolver C)

set(CMAKE_C_STANDARD 11)

add_executable(csolver
    src/main.c
    src/parse.c
    src/game.c
    src/solver.c
)

target_compile_options(csolver PRIVATE -O2 -Wall -Wextra -Wpedantic)

add_executable(trace
    src/trace.c
    src/parse.c
    src/game.c
)
target_compile_options(trace PRIVATE -O2 -Wall -Wextra -Wpedantic)

add_executable(csim
    src/csim.c
    src/parse.c
    src/game.c
    src/solver.c
)
target_compile_options(csim PRIVATE -O2 -Wall -Wextra -Wpedantic)
