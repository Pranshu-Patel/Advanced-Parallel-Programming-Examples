#include <stdio.h>
#include <string.h>
#include "mpi.h"

int main(int argc, char* argv[]) {
    int my_rank;
    int p;
    int source;
    int dest;
    int tag = 0;
    char message[100];
    int to_rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (my_rank != 0) {
        if (my_rank == p - 1)
            to_rank = 0;
        else
            to_rank = my_rank + 1;
        sprintf(message, "Hello from process %d to process %d!", my_rank, to_rank);

        dest = 0;
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    } else {
        for (source = p - 1; source >= 1; source--) {
            MPI_Recv(message, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
            printf("%s\n", message);
        }
    }

    MPI_Finalize();
    return 0;
}
