#include <stdio.h>
#include <stdlib.h>
#include<unistd.h>
#include "mpi.h"
int main(int argc, char* argv[])
{
    int my_rank;
    int p;
    int source;
    int dest;
    int tag=0;
    int i;
    int n=8;
    int *Arr;
    int j;
    MPI_Datatype derived;
    MPI_Status status;
    int buffer[100];
    int position;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Comm_size(MPI_COMM_WORLD, &p);

    Arr=(int*)malloc(n*n*sizeof(int));

    for(i=0;i<n*n;i++){
        Arr[i]=my_rank;
    }

    if(my_rank==0){
        position=1;
        printf("Matrix in process 0:\n");
        for(i=0;i<n;i++){
            MPI_Pack(&Arr[(i*n)+i],1,MPI_INT,buffer,100,&position,MPI_COMM_WORLD);
            for(j=0;j<n;j++){
                printf("%d ",Arr[(i*n)+j]);
            }
            printf("\n");
        }
        MPI_Send(buffer,100,MPI_PACKED,1,tag,MPI_COMM_WORLD);
    }
    else
    if(my_rank==1){
        position=0;
        sleep(1);
        printf("\nMatrix in process 1:\n");
        for(i=0;i<n;i++){
            for(j=0;j<n;j++){
                printf("%d ",Arr[(i*n)+j]);
            }
            printf("\n");
        }
        MPI_Recv(buffer,100,MPI_PACKED,0,tag,MPI_COMM_WORLD,&status);
        for(i=0;i<n;i++){
            MPI_Unpack(buffer,100,&position,&Arr[(i*n)+i],1,MPI_INT,MPI_COMM_WORLD);
        }
        printf("\nMatrix in process 1 after receiving diagonal from process 0:\n");
        for(i=0;i<n;i++){
            for(j=0;j<n;j++){
                printf("%d ",Arr[(i*n)+j]);
            }
            printf("\n");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
   
    free(Arr);

    MPI_Finalize();
    printf("\n");

    return 0;
}
