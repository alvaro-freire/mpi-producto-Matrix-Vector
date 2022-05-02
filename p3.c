#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>

#define DEBUG 0
#define N 6

int main(int argc, char *argv[]) {
    int i, j;
    struct timeval tc1, tc2, tm1, tm2;

    // variables for MPI
    int n_procs, rank;
    double total_pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int m = N / n_procs;
    int sendcounts[n_procs];    // array describing how many elements to send to each process
    int recvcounts[n_procs];    // array describing how many elements receiving from each process
    int displs[n_procs];        // array describing the displacements where each segment begins
    int rem = N % n_procs;      // elements remaining after division among processes
    int sum;                    // Sum of counts. Used to calculate displacements

    double matrix[N][N];
    double vector[N];
    double result[N];
    double *localresult;
    double localmatrix[N][N];

    if (rank == 0) {
        /* Initialize Matrix and Vector */
        for (i = 0; i < N; i++) {
            vector[i] = i;
            for (j = 0; j < N; j++) {
                matrix[i][j] = i + j;
            }
        }
    }

    /* measuring computation time */
    gettimeofday(&tc1, NULL);

    /* calculate send counts and displacements */
    sum = 0;
    for (int i = 0; i < n_procs; i++) {
        sendcounts[i] = 0;
        if (rem) {
            sendcounts[i]++;
            rem--;
        }
        sendcounts[i] = (N / n_procs + sendcounts[i]) * N;

        displs[i] = sum;
        sum += sendcounts[i];
    }

    int localrows = sendcounts[rank] / N;

    /* local memory allocation */
    localresult = malloc(sizeof(double) * localrows);
    // localmatrix = malloc(sizeof(double*) * localrows);
    // for (i = 0; i < localrows; i++)
    //     localmatrix[i] = (double *) malloc(sizeof(double) * N);

    /* divide the data among processes as described by sendcounts and displs */
    MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE, localmatrix, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vector, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* operations */
    for (i = 0; i < localrows; i++) {
        localresult[i] = 0;
        for (j = 0; j < N; j++) {
            localresult[i] += localmatrix[i][j] * vector[j];
        }
    }

    /* calculate receive counts and displacements */
    sum = 0;
    for (i = 0; i < n_procs; i++) {
        recvcounts[i] = sendcounts[i] / N;
        displs[i] = sum;
        sum += sendcounts[i] / N;
    }

    /* return the calculated data to root process as described by recvcounts and displs */
    MPI_Gatherv(localresult, localrows, MPI_DOUBLE, &result, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* measuring computation time */
    gettimeofday(&tc2, NULL);

    int microseconds = (tc2.tv_usec - tc1.tv_usec) + 1000000 * (tc2.tv_sec - tc1.tv_sec);

    /* Display result */
    if (DEBUG) {
        for (i = 0; i < N; i++) {
            printf(" %.2f \t ", result[i]);
        }
        printf("\n");
    } else {
        printf("Computation time of process %d (seconds) = %lf\n", rank, (double) microseconds / 1E6);
    }

    if (rank == 0) {
        sleep(1);

        /* matrix */
        printf("Matrix\n");
        for (i = 0; i < N; i++) {
            printf("[ ");
            for (j = 0; j < N; j++)
                printf("%.2f ", matrix[i][j]);
            printf("]\n");
        }
        printf("\n");

        /* vector */
        printf("Vector\n[ ");
        for (i = 0; i < N; i++)
            printf("%.2f ", vector[i]);
        printf("]\n\n");

        /* result */
        printf("Result\n[ ");
        for (i = 0; i < N; i++)
            printf("%.2f ", result[i]);
        printf("]\n\n");
    }

    free(localresult);
    //for (i = 0; i < localrows; i++)
    //    free(localmatrix[i]);
    //free(localmatrix);

    MPI_Finalize();

    return 0;
}
