#define PY_SSIZE_T_CLEAN
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "spkmeans.h"

void input_error(){

      printf("Invalid Input!");
      exit(1);
}

void print_elem(double e, int i, int N){
   
    (i!=N-1) ?
        (fabs(e)<0.0001) ? printf ("0.0000,") : printf("%.4f,",e)
    :
        (fabs(e)<0.0001) ? printf ("0.0000\n") : printf("%.4f\n", e);
}

void print_resault(double **A, double **B, double **df_c, int N, char *goal){
    int i,j;

    for(i=0;i<N+1;i++){
        for(j=0;j<N;j++){
            if(i==0){
                if(!strcmp(goal, "jacobi"))
                    print_elem(df_c[j][j], j, N);
                else continue;
            }
            else if(!strcmp(goal, "ddg"))
                print_elem(B[i-1][j], j, N);
            else
                print_elem(A[i-1][j], j, N);
        } 
    }
}

void free_matrix(double **matrix, int N){
    int i;

    for(i=0; i<N; i++)
        free(matrix[i]);
    free(matrix);
}

/* set C to A*B. | A,B,C:([N*N]) */
void matrix_mul(double **A, double **B, double **C, int N){
    double **D;
    int i,j,k;

    D = malloc_matrix(N,N);

    for(i=0; i<N; i++){ /* D = A*B */
        for(j=0; j<N; j++){
            D[i][j] = 0;
            for(k=0; k<N; k++)
                D[i][j]+= A[i][k]*B[k][j];
        }
    }

    for(i=0; i<N; i++) /* C = D */
        for(j=0; j<N; j++)
            C[i][j] = D[i][j];

    free_matrix(D,N);
}

/* set A to A transposed | A:([N*N]) */
void transpose(double** A, int N){
    int i,j;
    double tmp;

    for(i=0; i<N; i++){
        for(j=i+1; j<N; j++){
            tmp = A[i][j];
            A[i][j] = A[j][i];
            A[j][i] = tmp;
        }
    }
}

/* input: matrix A, and array IJ of size 2.
--> set IJ to the indices of the off-diagonal element in A with the largest 
absolute value, i.e A[IJ[0]][IJ[1]] = max{|A[i][j]|:i!=j}.  | A:([N*N]),IJ:([2]) */
void set_IJ(double **A, int *IJ, int N){
    double largest=-DBL_MAX;
    int i,j;

    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            if(i!=j && fabs(A[i][j])>largest){
                largest = fabs(A[i][j]);
                IJ[0]=i; IJ[1]=j;  
            }
        }
    }
}

/* set P to rotation matrix */
void set_P(double **P, double c, double s, int I, int J, int N){
    int i,j;

    for(i=0; i<N; i++){        
        for(j=0; j<N; j++){
            if(i==j)
                P[i][j] = (i==I||i==J) ? c : 1 ;
            else if(i==J && j==I)
                P[i][j]=-s;
            else P[i][j] = (i==I && j==J) ? s : 0;
        }
    }
}

double** malloc_matrix(int N, int K){
    double** A;
    int i;

    A = malloc(N*sizeof(double*));
    for(i=0; i<N; i++)
        A[i] = malloc(K*sizeof(double));

    return A;
}

/* set I to id matrix | I:([N*N]) */
void ID(double **I, int N){
    int i,j;

    for(i=0;i<N;i++)
        for(j=0;j<N;j++)
            I[i][j] = i==j ? 1 : 0;
}

/* return off-diagonal sum of squares of matrix A | A:([N*N]) */ 
double off(double** A, int N){
    double sum_sq=0;
    int i,j;

    for(i=0; i<N; i++)
        for(j=0; j<N; j++)
            sum_sq += i==j ? 0 : pow(A[i][j],2);

    return sum_sq;
}

/* return Euclidean Norm of X-Y | X,Y:([d]) */
double euclidean_norm_sub(double *X, double *Y, int d){
    double sum_square = 0;
    int i;

    for(i=0; i<d; i++)
        sum_square += pow(X[i]-Y[i],2);

    return sqrt(sum_square);
}

/* input: vector V and centroids matrix C.
--> return index i of the nearest centroid in C to V. | V:([d]), C:([k*d]) */
int min_dist(double *V, double **C, int k, int d){
    double min = DBL_MAX, dist;
    int i,  _index = -1; 

    for(i= 0;i<k; i++){
        dist = euclidean_norm_sub(V,C[i],d);
        if(dist<min){
            min = dist; 
            _index = i;
        }       
    }

    return _index;
}

/* input: matrix E with eigenvalues on the diagonal.
--> determine the number of clusters k. | E:([N*N]) */
int eigengap_heuristic(double **E, int N){
    int i, k=1;
    double d = 0;

    for (i=0; i<N/2; i++){
        if((fabs(E[i+1][i+1]-E[i][i])>d)){
            d = fabs(E[i+1][i+1]-E[i][i]);
            k = i+1;
        }
    }

    return k;
}

void d_swap(double **A, int i, int j){
    double *tmp;

    tmp = A[i];
    A[i] = A[j];
    A[j] = tmp;
}

void swap(double **A, int i, int j){
    double tmp;

    tmp = A[i][i];
    A[i][i] = A[j][j];
    A[j][j] = tmp;
}

/* sort diagonal of A in increasing order, and change B rows respectively,
i.e  B[i] is coordinated with A[i][i]  */
void sort_matrices(double **A, double **B, int l, int r){
   int i, j, pivot;

   if(l<r){
      pivot=l;
      i=l;
      j=r;
      while(i<j){
         while(A[i][i]<=A[pivot][pivot]&&i<r)
            i++;
         while(A[j][j]>A[pivot][pivot])
            j--;
         if(i<j){
            swap(A, i, j);
            d_swap(B, i, j);
         }
      }
      swap(A, pivot, j);
      d_swap(B, pivot, j);

      sort_matrices(A, B, l, j-1);
      sort_matrices(A, B, j+1, r);
   }
}

/* input: data matrix df, and matrix A. 
--> set A to weighted adjacency matrix of df | df:([N*d]),W:([N*N]) */
void wam(double** A, double** df, int N, int d){
    int i,j;

    for(i=0; i<N; i++)
        for(j=0; j<N; j++)
            A[i][j] = (i==j) ? 0 : exp(euclidean_norm_sub(df[i],df[j],d)/-2);    
}

/* input: weighted adjacency matrix A, and matrix B. 
--> set B to diagonal degree matrix of A | A,B:([N*N])  */
void ddg(double** B, double** A, int N){
    double sum;
    int i,j;

    for(i=0; i<N; i++){
        sum=0;
        for(j=0; j<N; j++){
            B[i][j] = 0;
            sum+=A[i][j];
        }
        B[i][i] = sum;
    }    
}

/* input: weighted adjacency matrix W, and diagonal degree matrix D.
--> set W to the normalized Laplacian matrix | W,D:([N*N]) */ 
void lnorm(double** W, double** D, int N){
    int i,j;

    for(i=0; i<N; i++) /* D = D^(-0.5) */
        D[i][i]=1/sqrt(D[i][i]);

    matrix_mul(D,W,W,N); /* W = D*W */
    matrix_mul(W,D,W,N); /* W = D*W*D */

    for(i=0; i<N; i++)
        for(j=0; j<N; j++)
            W[i][j] = (i==j) ? 1-W[i][j] : -W[i][j];
}

/* input: symmetric matrix A.
--> set A to eiganvalues of original symmetric A (eiganvalues on the diagonal) ,
--> set V to eiganvectors of original symmetric A (eiganvectors on columns). 
--> A[i][i] is corresponding to column i in V (= Vt[i]) | A,V:([N*N])  */ 
void jacobi(double** A, double** V, int N){
    double  **P, t;
    int *IJ, max_iter=100;

    P = malloc_matrix(N,N);   
    IJ = malloc(2*sizeof(int)); 
    ID(V,N); /* initialize V to identity matrix */

    while(max_iter>0){
        set_IJ(A, IJ, N); /* A[IJ[0]][IJ[1]] = largest off-diagonal element of A */
        t = (A[IJ[1]][IJ[1]] - A[IJ[0]][IJ[0]])/(2*A[IJ[0]][IJ[1]]); 
        if(t==0) t=1;
        else t=(t/fabs(t))/(fabs(t) + sqrt(1+t*t)); 
        set_P(P,1/sqrt(1+t*t),t/sqrt(1+t*t),IJ[0],IJ[1],N); /* P = rotation matrix */
        t = off(A, N); 
        transpose(P, N); 
        matrix_mul(P, A, A, N); /* A = Pt*A */
        transpose(P, N); 
        matrix_mul(A, P, A, N); /* A = Pt*A*P */
        if(t - off(A, N) < pow(10,-5)) /* convergence check */
             max_iter = 0;
        matrix_mul(V,P,V,N); /* V = V*P */
        max_iter--;
    }            

    free_matrix(P, N);
    free(IJ);
}
void exe_goal(double **df_c, double **A, double **B, int N, int d, char *goal){

    if(!strcmp(goal, "jacobi")){
        jacobi(df_c, A, N); /* df_c = eiganvalues, A = eiganvectors */
        return;
    }

    wam(A, df_c, N, d); /* A = weighted adjacency matrix of df_c*/
    if(!strcmp(goal, "wam")) 
        return;

    ddg(B, A, N); /* B = diagonal degree matrix of A */
    if(!strcmp(goal, "ddg")) 
        return; 

    lnorm(A, B, N); /* A = normalized Laplacian matrix of B */
    if(!strcmp(goal, "lnorm")) 
        return;

    if(!strcmp(goal, "spk")){
        jacobi(A, B, N); /* A = eiganvalues, B = eiganvectors */
        return;
    }

    input_error();
}

int main(int argc, char *argv[]){
    char *goal, *file_name, *delim =",", *ptr, line[1000];
    FILE *input_file = NULL;
    int N, d, i, j;
    double **df_c, **A, **B;
    
     
    if(argc != 3)
        input_error();

    goal = argv[1];
    file_name = argv[2];

    input_file = fopen(file_name, "r" );

    if(input_file==NULL || !strcmp(goal, "spk"))
        input_error();

    N = 0;
    d = 1;

    while(fgets(line, 1000, input_file)!=NULL){
        N++; 
        for (i=0; N < 2 && (unsigned)i<strlen(line)-1; i++) 
            if(line[i]==',')
                d++; 
    }
    
    df_c = malloc_matrix(N,d);
    input_file = fopen(file_name, "r" );

    for(i=0; i<N; i++){
        fgets(line, 1000, input_file);
        ptr = strtok(line, delim);
        for(j=0; ptr!=NULL ;j++){
            df_c[i][j] = atof(ptr);
            ptr = strtok(NULL, delim);
        }
    }

    fclose(input_file);

    /*********** spkmeans ************/

    A = malloc_matrix(N,N);
    B = malloc_matrix(N,N);

    exe_goal(df_c, A, B, N, d, goal);

    print_resault(A, B, df_c, N, goal); 
        
    free_matrix(A, N); 
    free_matrix(B, N);
    free_matrix(df_c, N); 

    return 0;
}


