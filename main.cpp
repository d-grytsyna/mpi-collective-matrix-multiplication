#include <mpi.h>
#include <iostream>
#define MASTER 0
#define N 100


double **alloc_2d(int rows, int cols) {
  double *m = (double *)malloc(rows * cols * sizeof(double));
  double **A = (double **)malloc(rows * sizeof(double *));

  A[0] = m;
  for (int i = 1; i < rows; i++) A[i] = A[i - 1] + cols;

  return A;
}

int check_matrix(double* matrix, int rows, int cols) {
    double first_element = matrix[0];
    for (int i = 0; i < rows * cols; i++) {
        if (matrix[i] != first_element) {
            return 0;
        }
    }
    return 1;
}

void print_results(char *prompt, double **a){
    int i, j;

    printf ("\n\n%s\n", prompt);
    for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%6.2f ", a[i][j]);
            }
            printf ("\n");
    }
    printf ("\n\n");
}

void matrix_multiplication_collective(int argc, char *argv[]){
    int numtasks,
    taskid,
    i, j, k, rc;
    double **a = alloc_2d(N, N);
    double **b = alloc_2d(N, N);
    double **c = alloc_2d(N, N);
    double **aa = alloc_2d(N, N);
    double **cc = alloc_2d(N, N);

    MPI_Init( &argc, &argv);
    
    MPI_Comm_size( MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank( MPI_COMM_WORLD, &taskid);

    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD,rc);
        exit(1);
    }
    double t1 = MPI_Wtime();
    if (taskid == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", numtasks);
        for (i=0; i<N; i++)
            for (j=0; j<N; j++)
                a[i][j]= 10;
        
        for (i=0; i<N; i++)
            for (j=0; j<N; j++)
                b[i][j]= 10;
    }
    MPI_Scatter(a[0], N*N/numtasks, MPI_DOUBLE, aa[0], N*N/numtasks, MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(b[0], N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Barrier(MPI_COMM_WORLD);

        for (i = 0; i < N / numtasks; i++) {
            for (k = 0; k < N; k++) {
                cc[i][k] = 0.0;
                for (j = 0; j < N; j++) {
                    cc[i][k] += aa[i][j] * b[j][k];
                }
            }
        }


    
//    MPI_Gather(cc[0], NRA*NCB/numtasks, MPI_DOUBLE, c[0], NRA*NCB/numtasks, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    
    
    MPI_Allgather(cc[0], N*N/numtasks, MPI_DOUBLE, c[0], N*N/numtasks, MPI_DOUBLE,  MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (taskid == MASTER){
        print_results("C = ", c);
        if (check_matrix(c[0], N, N)) {
            printf("All elements in the matrix C are the same.\n");
        } else {
            printf("Elements in the matrix C are not all the same.\n");
        }
    }
    MPI_Finalize();
    
    
}


void matrix_multiplication_collective_V(int argc, char *argv[]){
    int numtasks,
    taskid,
    i, j, k, rc;
    double **a = alloc_2d(N, N);
    double **b = alloc_2d(N, N);
    double **c = alloc_2d(N, N);
    double **aa = alloc_2d(N, N);
    double **cc = alloc_2d(N, N);
    MPI_Init( &argc, &argv);
    
    MPI_Comm_size( MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank( MPI_COMM_WORLD, &taskid);

    if (numtasks < 2 ) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD,rc);
        exit(1);
    }
   
    if (taskid == MASTER) {
        printf("mpi_mm has started with %d tasks.\n", numtasks);
        for (i=0; i<N; i++)
            for (j=0; j<N; j++)
                a[i][j]= 10;
        
        for (i=0; i<N; i++)
            for (j=0; j<N; j++)
                b[i][j]= 10;
    }
    double t1 = MPI_Wtime();
    int extra = N % numtasks;
    
    int *scounts = (int *)malloc(numtasks * sizeof(int));
    int *displs = (int *)malloc(numtasks * sizeof(int));


    for (int i = 0; i < numtasks; i++) {
        int divided_rows = N/numtasks;
        if (i < extra) {
            divided_rows++;
        }
        scounts[i] = divided_rows * N;
     
    }


    displs[0] = 0;
    for (int i = 1; i < numtasks; i++) {
      displs[i] = displs[i - 1] + scounts[i - 1];
    }

    MPI_Scatterv(a[0], scounts, displs, MPI_DOUBLE, aa[0], scounts[taskid], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(b[0], N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Barrier(MPI_COMM_WORLD);
    
    printf("Rank %d received %d rows:\n", taskid, scounts[taskid]/N);
//    for (int i = 0; i < scounts[taskid]/N; i++) {
//        for (int j = 0; j < N; j++) {
//            printf("%6.2f ", aa[i][j]);
//        }
//        printf("\n");
//    }

        for (i = 0; i < scounts[taskid]/N; i++) {
            for (k = 0; k < N; k++) {
                cc[i][k] = 0.0;
                for (j = 0; j < N; j++) {
                    cc[i][k] += aa[i][j] * b[j][k];
                }
            }
        }


    
    MPI_Gatherv(cc[0], scounts[taskid], MPI_DOUBLE, c[0], scounts, displs, MPI_DOUBLE, 0,MPI_COMM_WORLD);
    
//    MPI_Allgatherv(cc[0], scounts[taskid], MPI_DOUBLE, c[0], scounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (taskid == MASTER){
        t1 = MPI_Wtime() - t1;
        printf("\nExecution time: %.2f\n", t1);
//        print_results("C = ", c);
        if (check_matrix(c[0], N, N)) {
            printf("All elements in the matrix C are the same.\n");
        } else {
            printf("Elements in the matrix C are not all the same.\n");
        }
    }
    MPI_Finalize();
    
    
}


int main(int argc, char *argv[]) {

    matrix_multiplication_collective_V(argc, argv);
    return 0;
}
