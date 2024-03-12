#include "stdio.h"
#include "string.h"
#include "mpi.h"
#include "omp.h"

int main(int argc, char* argv[]) {
    int my_rank;
    int p;
    int source;
    int dest;
    int tag = 0;
    MPI_Status status;
    int broadcast_integer;
    int spacing;
    int stage;

    // Initialize MPI Communication
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    broadcast_integer = -1;
    if (my_rank == 0) broadcast_integer = 100;

    // Calculate the largest power of 2 less than or equal to the number of processes
    int largest_power_of_2;
    if (p > 1) {
        int posNum = 0;
        while ((1u << posNum) <= p) {
            posNum++;
        }
        largest_power_of_2 = 1u << (posNum - 1);
    } else {
        largest_power_of_2 = 1;
    }

    spacing = largest_power_of_2;
    stage = 0;

    while (spacing > 1) {
        #pragma omp parallel private(source, dest)
        {
            // Get the total number of threads and the thread's ID
            int num_threads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();

            // Calculate the portion of the loop each thread will execute
            int chunk_size = (p + num_threads - 1) / num_threads;
            int start = thread_id * chunk_size;
            int end = (thread_id + 1) * chunk_size;
            if (end > p) end = p;

            for (int i = start; i < end; i++) {
                // Check if the process is a dummy process
                if (i >= p) {
                    // Dummy process, no communication
                    MPI_Send(&broadcast_integer, 1, MPI_INT, MPI_PROC_NULL, 0, MPI_COMM_WORLD);
                } else if (i % spacing == 0) {
                    dest = i + spacing / 2;
                    if (dest < p) {
                        printf("Process %d, sending message to process %d at stage %d\n", my_rank, dest, stage);
                        MPI_Send(&broadcast_integer, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
                    }
                } else if (i % (spacing / 2) == 0) {
                    source = i - spacing / 2;
                    if (source >= 0) {
                        printf("Process %d, receive message from process %d at stage %d\n", my_rank, source, stage);
                        MPI_Recv(&broadcast_integer, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
                    }
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        spacing = spacing / 2;
        stage = stage + 1;
    }

    MPI_Finalize();
    printf("Process %d has integer %d\n", my_rank, broadcast_integer);
    return 0;
}
