# Selected image examples

## Baseline comparisons

|Dataset|Human and animal subjects|Vehicles and Objects|Misc|
|---|---|---|---|
|v1|2, 6, 7, 13, 52, 55, 62, 86|28, 30, 31, 32, 39, 95|45|
|v2|0, 8, 19, 20, 24, 62, 75, 81, 83, 92, 98|32, 33, 34, 37, |65|
|v3|12, 13, 16, 22, 25, 27, 30, 31, 33, 39, 45, 53, 55, 59, 64, 86, 93, 98|21, 37, 50, 76, 79, 81, 74, 83, 88||

- pick the best ones and make bigger (4 rows)
    - 2: 19, 75, 
    - 3: 16, 86

## Foveation patterns
|Figure|Image|
|---|---|
|Radius||
|Circular||
|Multi-circle||

## Saliency
|Figure|Image|
|---|---|
|Single object|3, 5, 17, 26, 27, 30, 31, 32, 33, 35, 37, 40, 42, 43, 44, 45, 47, 59, 61, 64, 67, 68, 69, 71, 76, 82, 84, 95, 99|
|Multiple objects|0, 1, 4, 25, 40, 47, 61, 86, 88|
|Comparison with Random Masking|5, 17|
|Games|1, 4, 24, 25, 28, 29, 32, 34, 41, 45, 53, 54, 55, 62, 66, 69, 75, 77, 82, 86, 87, 90, 99|
|Robotics|0, 4, 15, 16, 25, 32, 33, 40, 42, 50, 56, 57, 65, 66, 67, 75, 78, 79, 83, 90|
|Self-driving cars|0, 1, 6, 7, 13, 18, 20, 29, 34, 48, 53, 58, 64, 70, 72, 79, 85, 94, 96, 97|

- single object: 5, 26, 27, 30, 32, 33, 35, 37, 40, 43, 45, 47, 61, 64, 68, 71, 76, 84, 95, 99

## Bounding-box
|Figure|Image|
|---|---|
|Single object|17, 23, 26, 27, 31, 33, 40, 41, 42, 43, 44, 45, 47, 64, 69, 70, 76, 80, 81, 82, 83, 84, 90, 95, 99|
|Multiple objects|0, 3, 4, 10, 19, 23, 24, 25, 29, 32, 38, 49, 55, 56, 61, 86, 88, 90, 91, 95, 98, 99|
|Comparison with Saliency||