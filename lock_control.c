#include <stdio.h>
#include <omp.h>

#define NUM_WORKERS 4

omp_lock_t locks[NUM_WORKERS]; // Locks for each worker thread
int busy_states[NUM_WORKERS] = {0}; // 0 represents idle, 1 represents busy

void worker(int tid) {
    while (1) {
        // Simulate some work
        printf("Worker %d is doing some work.\n", tid);
        for (int i = 0; i < 10000000; i++) {} // Delay loop
        
        // Update busy state using locks
        omp_set_lock(&locks[tid]);
        busy_states[tid] = 0; // Worker becomes idle after completing the work
        omp_unset_lock(&locks[tid]);
    }
}

int main() {
    // Initialize locks
    for (int i = 0; i < NUM_WORKERS; i++) {
        omp_init_lock(&locks[i]);
    }

    // Start worker threads
    #pragma omp parallel num_threads(NUM_WORKERS + 1)
    {
        int tid = omp_get_thread_num();

        if (tid == 0) {
            // Master thread
            while (1) {
                // Check for idle workers and assign work
                for (int i = 1; i <= NUM_WORKERS; i++) {
                    omp_set_lock(&locks[i - 1]);
                    if (busy_states[i - 1] == 0) {
                        printf("Assigning work to Worker %d.\n", i);
                        busy_states[i - 1] = 1; // Worker becomes busy
                        omp_unset_lock(&locks[i - 1]);
                        break; // Work assigned, break the loop
                    }
                    omp_unset_lock(&locks[i - 1]);
                }
                // Simulate some time before checking again
                for (int i = 0; i < 10000000; i++) {} // Delay loop
            }
        } else {
            // Worker thread
            worker(tid - 1);
        }
    }

    // Destroy locks
    for (int i = 0; i < NUM_WORKERS; i++) {
        omp_destroy_lock(&locks[i]);
    }

    return 0;
}
