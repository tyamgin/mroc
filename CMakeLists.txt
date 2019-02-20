cmake_minimum_required(VERSION 3.13)
project(mroc C)

set(CMAKE_C_STANDARD 11)

add_executable(mroc main.c roc.c)