#include "dnf_executorch.h"

#define _USE_MATH_DEFINES

#include <math.h>

#include <stdio.h>

// Testing if the DNF is able to learn to eliminate
// a cosine signal with a sine signal!

void doTest(const char* filename, float failAbove, int maxSteps)
{
    double mu = 0.1;

    DNF_executorch dnf(filename, mu);
    dnf.setLearning(true);

    for (int i = 0; i < maxSteps; i++)
    {
        const double input_signal = cos(2 * M_PI / 20 * i) * 0.1;
        const double ref_noise = sin(2 * M_PI / 20 * i) * 0.1;
        const double output_signal = dnf.filter(input_signal, ref_noise);
        printf("%lf %lf %lf\n", input_signal, ref_noise, output_signal);
        if ((i > (maxSteps-50)) && (fabs(output_signal) > failAbove))
        {
            fprintf(stderr, "LMS did not cancel the sine wave with the cosine wave.\n");
            throw;
        }
    }
}


int main(int, char **)
{
    doTest("dnf_executorch1.pte",0.0005,500);
    doTest("dnf_executorch5.pte",0.005,1000);
}
