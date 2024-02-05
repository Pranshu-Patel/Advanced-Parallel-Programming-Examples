#include <stdio.h>
#include <mpi.h>


#define N 4 // Matrix size


void printMatrix(int matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);


    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int matrix[N][N];


    // Initialize matrices
    if (rank == 0) {
        printf("Process 0 initializing matrix with 2s\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = 2;
            }
        }
    } else if (rank == 1) {
        printf("Process 1 initializing matrix with 2s\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = 1;
            }
        }
    }


    // Printing initial matrices
    printf("Process %d initial matrix:\n", rank);
    printMatrix(matrix);


    MPI_Datatype diag_type;
    
    int blocklengths[N];
    MPI_Aint displacements[N];
    MPI_Datatype types[N];


    for (int i = 0; i < N; i++) {
        blocklengths[i] = 1;
        displacements[i] = i * sizeof(int);
        types[i] = MPI_INT;
    }


    MPI_Type_create_hindexed(N, blocklengths, displacements, MPI_INT, &diag_type);
    MPI_Type_commit(&diag_type);


    // Exchanging diagonal elements using MPI communication
    int send_buf[N], recv_buf[N];
    if (rank == 0) {
        printf("Process 0 sending matrix to Process 1\n");
        MPI_Send(matrix, 1, diag_type, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 receiving matrix from Process 1\n");
        MPI_Recv(recv_buf, 1, diag_type, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Updating the diagonal elements in the matrix
        for (int i = 0; i < N; i++) {
            matrix[i][i] = recv_buf[i];
        }
    } else if (rank == 1) {
        printf("Process 1 receiving matrix from Process 0\n");
        MPI_Recv(recv_buf, 1, diag_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Updating the diagonal elements in the matrix
        for (int i = 0; i < N; i++) {
            matrix[i][i] = recv_buf[i];
        }
        printf("Process 1 sending matrix to Process 0\n");
        MPI_Send(matrix, 1, diag_type, 0, 0, MPI_COMM_WORLD);
    }


    // Printing Process 1 matrix after diagonal exchange
    if (rank == 1) {
        printf("Process 1 matrix after diagonal exchange:\n");
        printMatrix(matrix);
    }


    MPI_Type_free(&diag_type);
    MPI_Finalize();
    return 0;
}
