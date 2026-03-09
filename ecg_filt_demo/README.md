# 50Hz filter

Demo which filters out 50Hz mains from a noisy ECG.
This is a minimal example where the reference noise is just
a 50Hz sine wave and the DNF removes the 50Hz from the
ECG by using this 50Hz sine wave. This, of course, can also be
done by a simple LMS FIR filter but it serves here as an example
which can then be used for more complex reference signals.

## How to run

```
./ecg_filt_demo
```

## How to plot the results?

```
gnuplot
plot "ecg_filtered.dat" using 1 w l
```

which plots the filtered ECG with 50Hz removed from 1sec onwards:

![alt tag](screenshot.png)
