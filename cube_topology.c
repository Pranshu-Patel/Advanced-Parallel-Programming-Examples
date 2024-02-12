#include <stdio.h>

#include <mpi.h>

 

int main(int argc, char *argv[]) {

    int rank, size;

    int dims[3] = {2, 2, 2};

    int periods[3] = {0, 0, 0};

    int coords[3];

    int sums[3] = {0, 0, 0};

 

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

 

    if (size != 8) {

        fprintf(stderr, "This program requires exactly 8 processes\n");

        MPI_Abort(MPI_COMM_WORLD, 1);

    }

 

    MPI_Comm comm_3d;

    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &comm_3d);

    MPI_Cart_coords(comm_3d, rank, 3, coords);

 

    for (int i = 0; i < 3; i++) {

        MPI_Comm side_comm;

        int side_coords[3] = {coords[0], coords[1], coords[2]};

       

        for (int j = 0; j < 2; j++) {

            side_coords[i] = j; // Vary one dimension to form a side

            int side_rank;

            MPI_Cart_rank(comm_3d, side_coords, &side_rank);

 

            MPI_Comm_split(comm_3d, side_rank, rank, &side_comm);

 

           

            int local_number = rank + 1;

            int side_sum;

            MPI_Reduce(&local_number, &side_sum, 1, MPI_INT, MPI_SUM, 0, side_comm);

 

            // Add the side sum to the total sum for the current side of the cube

            sums[i] += side_sum;

 

            MPI_Comm_free(&side_comm);

        }

    }

 

   printf("Process %d: Sums - X: %d, Y: %d, Z: %d\n", rank, sums[0], sums[1], sums[2]);

 

    MPI_Comm_free(&comm_3d);

    MPI_Finalize();

 

    return 0;

}