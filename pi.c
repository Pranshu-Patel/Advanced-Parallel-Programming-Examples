#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "mpi.h"

#define SEED 12345
#define NUM_POINTS 1000000

double monteCarloPi(int numPoints, int seed) {
    int pointsInsideCircle = 0;
    srand(seed);

    for (int i = 0; i < numPoints; ++i) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;

        if (x * x + y * y <= 1.0) {
            pointsInsideCircle++;
        }
    }

    return 4.0 * pointsInsideCircle / numPoints;
}

int main(int argc, char *argv[]) {
    int my_rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double start_time;
    if (my_rank == 0) {
        start_time = MPI_Wtime();
    }

    double local_pi = monteCarloPi(NUM_POINTS, SEED + my_rank);

    double *all_pi_values = NULL;
    if (my_rank == 0) {
        all_pi_values = (double *)malloc(num_procs * sizeof(double));
    }
    
    MPI_Gather(&local_pi, 1, MPI_DOUBLE, all_pi_values, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        double sum_pi = 0.0;
        for (int i = 0; i < num_procs; ++i) {
            sum_pi += all_pi_values[i];
        }

        double average_pi = sum_pi / num_procs;

        double end_time = MPI_Wtime();
        double execution_time = end_time - start_time;

        printf("Estimated value of Pi: %f\n", average_pi);
        printf("Execution time: %f seconds\n", execution_time);

        free(all_pi_values);
    }

    MPI_Finalize();
    return 0;
}
