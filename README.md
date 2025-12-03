```pseudocode
for point in image:
  int indicator[point][circle];
  for circle in circles:
    calculateIndicator(point,circles,indicator) # PARALLELIZED in both points x circles

for indicator for particular point:
  Indices[point] = calculateIndicesArray(Indicator) # using same technique as used in problem 2: use prefix sums of indicator array to calculate positiions

for each point in points:
  Multiply all elements in Indices[point] # use same implementation as prefix sums, replace int sum with float prod
```
