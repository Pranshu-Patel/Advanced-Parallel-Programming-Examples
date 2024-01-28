#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            printf("Please run the program with exactly 2 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    int message_size = 1; // Initial message size
    int *message;

    for (message_size = 1; message_size <= 4096; message_size *= 2) {
        MPI_Request request;
        MPI_Status status;

        if (rank == 0) {
            // Process 0 sends a message to process 1
            message = (int *)malloc(message_size * sizeof(int));
            MPI_Isend(message, message_size, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
            free(message);
            MPI_Wait(&request, &status);
        } else if (rank == 1) {
            message = (int *)malloc((message_size / 2) * sizeof(int));
            MPI_Irecv(message, message_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
            free(message);
            MPI_Wait(&request, &status);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
