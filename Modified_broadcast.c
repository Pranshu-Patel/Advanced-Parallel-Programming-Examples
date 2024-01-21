#include "stdio.h"
#include "mpi.h"
#include "math.h"

int main(int argc, char** argv){
    int my_rank;       /* rank of process      */
    int p;             /* number of processes  */
    int source;        /* rank of sender       */
    int dest;          /* rank of receiver     */
    int tag = 0;       /* tag for messages     */
    int broadcast_integer = -1; /* integer for broadcast */
    int spacing, stage = 0;      /* stage of broadcast */
    double fraction = 0.0; /* mantissa */ 
    int exp = 0;      /* fraction and exponent of 2*/
    MPI_Status  status;        /* status for receive   */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    broadcast_integer = (my_rank == 0) ? 100 : -1;

    // ceiling to the nearest power of 2
    fraction = frexp(p, &exp);
    spacing = pow(2, exp);

    while (spacing > 1){
        if (my_rank % spacing == 0)
        {
            dest = my_rank + spacing/2;
            dest = (dest >= p) ? MPI_PROC_NULL : dest;
            printf("%d sends to %d, stage: %d \n", my_rank, dest, stage);
            MPI_Send(&broadcast_integer, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        }
        else if (my_rank % (spacing/2) == 0)
        {
            source = my_rank - spacing/2;
            printf("%d receives from %d, stage: %d \n", my_rank, source, stage);
            MPI_Recv(&broadcast_integer, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status); 
        }

        spacing /= 2;
        stage++;
    }

    MPI_Finalize();

    return 0;
}
