```python
for point in image:
  int indicator[point][circle];
  for circle in circles:
    calculateIndicator(point,circles,indicator) # PARALLELIZED in both points x circles

for indicator for particular point:
  Indices[point] = calculateIndicesArray(Indicator) # using same technique as used in problem 2: use prefix sums of indicator array to calculate positiions

for each point in points:
  Multiply all elements in Indices[point] # use same implementation as prefix sums, replace int sum with float prod
```

```cpp
void inclusive_scan_iterative(int* start, int* end, int* output) {

    int N = end - start;
    memmove(output, start, N*sizeof(int));

    // upsweep phase
    for (int two_d = 1; two_d <= N/2; two_d*=2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            output[i+two_dplus1-1] += output[i+two_d-1];
        }
    }

    output[N-1] = 0;

    // downsweep phase
    for (int two_d = N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        parallel_for (int i = 0; i < N; i += two_dplus1) {
            int t = output[i+two_d-1];
            output[i+two_d-1] = output[i+two_dplus1-1];
            output[i+two_dplus1-1] += t;
        }
    }
}
```