#include <stdio.h>
#include <mpi.h>

#define MSG_LENGTH 1024

int main(int argc, char** argv)
{
	int my_rank;
	int p;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if(p < 3)
	{
		printf("Please run with at least 3 processes\n");
		MPI_Finalize();
		return 0;
	}
	
	int msg[MSG_LENGTH] = {111 * (my_rank + 1)};
	int r_msg_1[MSG_LENGTH], r_msg_2[MSG_LENGTH];

	int source = my_rank;
	int dest_1 = (my_rank + 1) % p; // to keep it withing 0 to p-1
	int dest_2 = (my_rank + p - 1) % p; // incrementing with p then -1 to keep it above zero
	
	// it's working for small msg lengths
	MPI_Send(msg, MSG_LENGTH, MPI_INT, dest_1, my_rank, MPI_COMM_WORLD);
	MPI_Send(msg, MSG_LENGTH, MPI_INT, dest_2, my_rank, MPI_COMM_WORLD);
	MPI_Recv(r_msg_1, MSG_LENGTH, MPI_INT, dest_1, dest_1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(r_msg_2, MSG_LENGTH, MPI_INT, dest_2, dest_2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	printf("process %d sent to processes %d & %d\n", my_rank, dest_1, dest_2);
	printf("process %d recevied from process %d\n", my_rank, dest_1);
	printf("process %d recevied from process %d\n", my_rank, dest_2);

	MPI_Finalize();
	return 0;
}
