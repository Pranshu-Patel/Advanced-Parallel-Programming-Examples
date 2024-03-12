#include <iostream>
#include <omp.h>

#define SIZE 1000000

int main() {
    int array[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        array[i] = i + 1; // Initialize array with values 1 to SIZE
    }

    int sum = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < SIZE; ++i) {
                #pragma omp task firstprivate(i)
                {
                    #pragma omp atomic
                    sum += array[i];
                }
            }
        }
    }

    std::cout << "Sum: " << sum << std::endl;

    return 0;
}
