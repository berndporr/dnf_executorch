/**
 * @file test_load_model.cpp
 * @author Bernd Porr
 * @brief Constructs the DNF and loads the model
 * @copyright Copyright (c) 2026
 * 
 */

#include <chrono>
#include <iostream>
#include <stdio.h>
#include <thread>

#define _USE_MATH_DEFINES
#include <math.h>

#include "dnf_executorch.h"

constexpr int expectedNTaps = 50;

int main(int argc, char *argv[])
{
    DNF_executorch dnf("dnf_executorch1.pte");
    if (dnf.getNumberOfTaps() != expectedNTaps) {
        fprintf(stderr, "Number of taps mismatch.\n");
        throw;
    }

    if (dnf.getSignalDelaySteps() != (expectedNTaps/2)) {
        fprintf(stderr, "Number of signal delay steps mismatch.\n");
        throw;
    }
}
