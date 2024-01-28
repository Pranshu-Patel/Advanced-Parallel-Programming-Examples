#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Controller process
    if (rank == 0) {
        for (int dest = 1; dest < size; dest++) {
            int idle_flag;
            MPI_Request request;
            MPI_Irecv(&idle_flag, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, &request);

            int flag = 0;
            while (!flag) {
                MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
                printf("Controller: Testing idle flag for process %d\n", dest);
            }

            if (idle_flag) {
                printf("Controller: Sending work to process %d\n", dest);
            }
        }
    }
    // Other processes
    else {
        int idle_flag = 1;
        printf("Process %d: Signaling idle to controller\n", rank);
        MPI_Send(&idle_flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    }

    MPI_Finalize();
    return 0;
}
