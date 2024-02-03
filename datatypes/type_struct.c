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

    // Create a struct for diagonal entries
    MPI_Datatype diagonalType;
    MPI_Type_vector(1, 1, 5, MPI_INT, &diagonalType);
    MPI_Type_commit(&diagonalType);

    // Send diagonal entries to another process
    if (rank == 0) {
        printf("Sending diagonal entry\n");
        MPI_Send(&matrix[0][0], 1, diagonalType, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        int receivedDiagonal;
        printf("Receiving diagonal entry\n");
        MPI_Recv(&receivedDiagonal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Process 1 now has the diagonal entry
    }

    MPI_Type_free(&diagonalType);
    MPI_Finalize();
    return 0;
}
