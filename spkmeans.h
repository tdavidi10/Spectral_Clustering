#ifndef SPKMEANS_H_
#define SPKMEANS_H_

void wam(double** A, double** df, int N, int d);
void ddg(double** B, double** A, int N);
void lnorm(double** W, double** D, int N);
void jacobi(double** A, double** V, int N);
void input_error();
void print_elem(double e, int i, int N);
void free_matrix(double **matrix, int N);
void transpose(double** A, int N);
void matrix_mul(double **A, double **B, double **D, int N);
void set_IJ(double **A, int *IJ, int N);
void set_P(double **P, double c, double s, int I, int J, int N);
void ID(double **I, int N);
void d_swap(double **A, int i, int j);
void swap(double **A, int i, int j);
void sort_matrices(double **A, double **B, int l, int r);
void exe_goal(double **df_c, double **A, double **B, int N, int d, char *goal);
void print_resault(double **A, double **B, double **df_c, int N, char *goal);
double** malloc_matrix(int N, int K);
double off(double **A, int N);
double euclidean_norm_sub(double *X, double *Y, int d);
int min_dist(double *vector, double **centroids, int k, int d);
int eigengap_heuristic(double **E, int N);
int main();

#endif