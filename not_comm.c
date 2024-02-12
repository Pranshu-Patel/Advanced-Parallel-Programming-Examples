#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the number of processes is at least 2
    if (size < 2) {
        printf("This example requires at least 2 processes.\n");
    } else {
        // Create communicators for consecutive pairs
        MPI_Comm pair_comm;
        MPI_Comm_split(MPI_COMM_WORLD, rank / 2, rank, &pair_comm);

        // Perform communication from process 0 to process N-1
        if (rank == 0) {
            char message[] = "Hello from process 0!";
            MPI_Send(message, sizeof(message), MPI_CHAR, size - 1, 0, pair_comm);
            printf("Process 0 sent the message: '%s' to process %d\n", message, size - 1);
        } else if (rank == size - 1) {
            char message[100];  // Adjust the buffer size accordingly
            MPI_Recv(message, sizeof(message), MPI_CHAR, 0, 0, pair_comm, MPI_STATUS_IGNORE);
            printf("Process %d received the message: '%s'\n", rank, message);
        } else {
            // Intermediate processes
        }

        // Clean up
        MPI_Comm_free(&pair_comm);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
