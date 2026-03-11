#include "dnf_executorch.h"

#define _USE_MATH_DEFINES

#include <math.h>

#include <stdio.h>

// Testing if the DNF is able to learn to eliminate
// a cosine signal with a sine signal!

double mu = 0.1;

int main(int, char **)
{
    DNF_executorch dnf("dnf_executorch.pte", mu);
    dnf.setLearning(true);

    for (int i = 0; i < 500; i++)
    {
        const double input_signal = cos(2 * M_PI / 20 * i) * 0.1;
        const double ref_noise = sin(2 * M_PI / 20 * i) * 0.1;
        const double output_signal = dnf.filter(input_signal, ref_noise);
        printf("%lf %lf %lf\n", input_signal, ref_noise, output_signal);
        if ( (i > 450) && (fabs(output_signal) > 0.0005) )
        {
            fprintf(stderr, "LMS did not cancel the sine wave with the cosine wave.\n");
            throw;
        }
    }
}
