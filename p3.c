#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <mpi.h>

#define DEBUG 1
#define N 6

int main(int argc, char *argv[]) {
    int i, j;
    struct timeval  tc1, tc2;
    struct timeval  tm1_s, tm2_s,
                    tm1_g, tm2_g,
                    tm1_b, tm2_b;

    /* variables for MPI */
    int n_procs, rank;
    double total_pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int sendcounts[n_procs];    // array describing how many elements to send to each process
    int recvcounts[n_procs];    // array describing how many elements receiving from each process
    int displs[n_procs];        // array describing the displacements where each segment begins
    int rem = N % n_procs;      // elements remaining after division among processes
    int sum;                    // Sum of counts. Used to calculate displacements

    double matrix[N][N];
    double vector[N];
    double result[N];

    if (rank == 0) {
        /* Initialize Matrix and Vector */
        for (i = 0; i < N; i++) {
            vector[i] = i;
            for (j = 0; j < N; j++) {
                matrix[i][j] = i + j;
            }
        }
    }

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

    /* local variables */
    int localrows = sendcounts[rank] / N;
    double localresult[localrows];
    double localmatrix[localrows][N];

    /* measuring communication time */
    gettimeofday(&tm1_s, NULL);
    /* divide the data among processes as described by sendcounts and displs */
    MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE, localmatrix, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* measuring communication time */
    gettimeofday(&tm2_s, NULL);

    /* measuring communication time */
    gettimeofday(&tm1_b, NULL);
    /* divide the vector data among processes */
    MPI_Bcast(&vector, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* measuring communication time */
    gettimeofday(&tm2_b, NULL);

    /* measuring computation time */
    gettimeofday(&tc1, NULL);
    /* operations */
    for (i = 0; i < localrows; i++) {
        localresult[i] = 0;
        for (j = 0; j < N; j++) {
            localresult[i] += localmatrix[i][j] * vector[j];
        }
    }
    /* measuring computation time */
    gettimeofday(&tc2, NULL);

    if (DEBUG) {
        /* local matrix */
        printf("Local matrix(%d)\n", rank);
        for (i = 0; i < localrows; i++) {
            printf("[ ");
            for (j = 0; j < N; j++)
                printf("%.2f ", localmatrix[i][j]);
            printf("]\n");
        }
        printf("\n");

        /* local result */
        printf("Local result(%d)\n[ ", rank);
        for (i = 0; i < localrows; i++)
            printf("%.2f ", localresult[i]);
        printf("]\n\n");
    }

    /* calculate receive counts and displacements */
    sum = 0;
    for (i = 0; i < n_procs; i++) {
        recvcounts[i] = sendcounts[i] / N;
        displs[i] = sum;
        sum += recvcounts[i];
    }

    /* measuring communication time */
    gettimeofday(&tm1_g, NULL);
    /* return the calculated data to root process as described by recvcounts and displs */
    MPI_Gatherv(&localresult, localrows, MPI_DOUBLE, &result, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* measuring communication time */
    gettimeofday(&tm2_g, NULL);

    int comptime = (tc2.tv_usec - tc1.tv_usec) + 1000000 * (tc2.tv_sec - tc1.tv_sec);
    int commtime_s = (tm2_s.tv_usec - tm1_s.tv_usec) + 1000000 * (tm2_s.tv_sec - tm1_s.tv_sec);
    int commtime_b = (tm2_b.tv_usec - tm1_b.tv_usec) + 1000000 * (tm2_b.tv_sec - tm1_b.tv_sec);
    int commtime_g = (tm2_g.tv_usec - tm1_g.tv_usec) + 1000000 * (tm2_g.tv_sec - tm1_g.tv_sec);
    int commtime = (commtime_s + commtime_b + commtime_g) / 3;

    /* Display result */
    if (DEBUG) {
        if (rank == 0) {

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
            printf("\n");
        }
    } else {
        printf("Process %d - Computation time(useconds) = %.2lf\n", rank, (double) comptime);
        //printf("Process %d - Communication time(scatter) = %.2lf\n", rank, (double) commtime_s);
        //printf("Process %d - Communication time(bcast) = %.2lf\n", rank, (double) commtime_b);
        //printf("Process %d - Communication time(gather) = %.2lf\n", rank, (double) commtime_g);
        printf("Process %d - Communication time(useconds) = %.2lf\n", rank, (double) commtime);
    }

    MPI_Finalize();

    return 0;
}
