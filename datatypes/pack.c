#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create a square matrix (for example, 4x4)
    int matrix[4][4];
    // Initialize matrix with values

    // Calculate the size needed for packing
    int packSize;
    MPI_Pack_size(4, MPI_INT, MPI_COMM_WORLD, &packSize);

    // Create a buffer for packing
    char buffer[packSize];

    // Pack the diagonal entries into the buffer
    if (rank == 0) {
        printf("Sending diagonal entry\n");
        MPI_Pack(&matrix[0][0], 1, MPI_INT, buffer, packSize, &packSize, MPI_COMM_WORLD);
        MPI_Send(buffer, packSize, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        printf("Receiving diagonal entry\n");
        MPI_Recv(buffer, packSize, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int receivedDiagonal;
        int position = 0;
        MPI_Unpack(buffer, packSize, &position, &receivedDiagonal, 1, MPI_INT, MPI_COMM_WORLD);
        // Process 1 now has the diagonal entry
    }

    MPI_Finalize();
    return 0;
}
