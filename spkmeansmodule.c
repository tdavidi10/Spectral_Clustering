#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h> 
#include <string.h>
#include "spkmeans.h"

/* input: PyObject final_matrix, matrix A, matrix B.
if(B!=NULL): set final_matrix to B with the diagonal of A as the  
             first row --> final_matrix:([(N+1)*N]). | k == N,  A,B:([N*N])
else: set final_matrix to A --> final_matrix:([N*K]). | A:([N*K]) */
void set_matrix(PyObject* final_matrix, double **A, double **B, int N, int k){
    int i,j;
    PyObject* rows;

    for(i=0; i<N+(B!=NULL); i++){
        rows = PyList_New(k);
        for(j=0; j<k; j++){
            if(B!=NULL)
                (i==0) ? PyList_SetItem(rows, j, Py_BuildValue("d", A[j][j]))
                       : PyList_SetItem(rows, j, Py_BuildValue("d", B[i-1][j]));
            else PyList_SetItem(rows, j, Py_BuildValue("d", A[i][j]));
        }
        PyList_SetItem(final_matrix, i, Py_BuildValue("O", rows));
    }
}

void set_T(PyObject* T, double** A, double** B, int N, int k) {
    int i,j,l;
    double sum_sq;

    transpose(B, N);   
    sort_matrices(A ,B, 0, N-1);
    transpose(B, N);

    if (k==0)
        k = eigengap_heuristic(A, N);
    
    for(i=0; i<N; i++){
        for(j=0; j<k; j++){
            sum_sq = 0;
            for (l=0; l<k; l++)
                sum_sq+=B[i][l]*B[i][l];
            A[i][j] = (sum_sq==0) ? 0 : B[i][j]/sqrt(sum_sq);
        }
    }
    
    set_matrix(T, A, NULL ,N, k);    
}

void set_final_martix(PyObject* df_py, double **df_c, double **A, double **B, int N, int k, char *goal){
    
    if(!strcmp(goal,"spk"))
        set_T(df_py, A, B, N, k);

    else if(!strcmp(goal, "jacobi"))
        set_matrix(df_py, df_c, A, N, N);

    else if(!strcmp(goal, "ddg"))
        set_matrix(df_py, B, NULL, N, N); 

    else
        set_matrix(df_py, A, NULL, N, N); 
}

/* set R with the data in py_ob. | R:([N*k]) */
void set_data( double**R, PyObject* py_ob, int N, int k){
    int i,j;

    for(i=0; i<N; i++)
        for(j=0; j<k; j++) 
            R[i][j] = PyFloat_AsDouble(
                      PyList_GetItem(
                      PyList_GetItem(py_ob, i), j));
}


PyObject* kmeans(PyObject* vectors_py, PyObject* centroids_py, int d){
    int N, k, i, j, nearest_cent, max_iter=300, *cluster_size;
    double **centroids, **vectors, **weights, **cluster_weight;

    N = PyList_Size(vectors_py);
    k = PyList_Size(centroids_py);

    weights = malloc_matrix(k,d);
    cluster_weight = malloc_matrix(k,d);
    cluster_size = malloc(k*sizeof(int));
    vectors = malloc_matrix(N, d);
    centroids = malloc_matrix(k, d); 

    set_data(vectors, vectors_py, N, d);
    set_data(centroids, centroids_py, k, d);

    vectors_py =  PyList_New(k);

    while(max_iter>0){
        for(i=0; i<k; i++){ 
            cluster_size[i]=0;
            for(j=0; j<d; j++)
                cluster_weight[i][j] = 0;
        }
        for(i=0; i<N; i++){
            nearest_cent = min_dist(vectors[i], centroids, k, d);
            cluster_size[nearest_cent]++; 
            for(j=0; j<d;j++)
                cluster_weight[nearest_cent][j] += vectors[i][j]; 
        }
        for(i=0; i<k; i++){
            for(j=0; j<d; j++)
                weights[i][j] = (cluster_size[i]==0) ? 0 : cluster_weight[i][j]/cluster_size[i]; 
            if(euclidean_norm_sub(centroids[i], weights[i], d) < 0) 
                max_iter=0;
            for(j=0; j<d; j++)
                centroids[i][j]=weights[i][j];                  
        }
        max_iter--;
    } 

    set_matrix(vectors_py, centroids, NULL ,k, d); 

    free_matrix(centroids, k);
    free_matrix(vectors, N);
    free_matrix(weights, k);
    free_matrix(cluster_weight, k);
    free(cluster_size);

    return vectors_py;
}

static PyObject* fit(PyObject* self, PyObject* args){
    PyObject *df_py, *centroids_py, *goal_py;
    int d, N, k;
    char *goal;
    double **df_c, **A, **B;

    PyArg_ParseTuple(args, "OOOii", &df_py, &centroids_py, &goal_py, &d, &k); 

    goal = strtok(PyBytes_AS_STRING(
               PyUnicode_AsEncodedString(
               PyObject_Repr(goal_py), "utf-8", "~E~")), "'");

    if(!strcmp(goal, "kmeans")) 
        return kmeans(df_py, centroids_py, d);

    N = PyList_Size(df_py);
    A = malloc_matrix(N,N); 
    B = malloc_matrix(N,N);
    df_c = malloc_matrix(N, d);

    set_data(df_c, df_py, N, d);
    exe_goal(df_c, A, B, N, d, goal);
    
    df_py = PyList_New(N+(!strcmp(goal,"jacobi")?1:0)); /* for jacobi we need N+1 rows */

    set_final_martix(df_py, df_c, A, B, N, k, goal);

    free_matrix(A, N);
    free_matrix(B, N);
    free_matrix(df_c, N);
    
    return df_py;
}

/* C-Python API */
static PyMethodDef spkmeansMethods[] = {
        {"fit", (PyCFunction) fit, METH_VARARGS, PyDoc_STR("C spkmeans algorithm")},
        {NULL, NULL, 0, NULL}
};
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "myspkmeanssp", NULL, -1, spkmeansMethods
};
PyMODINIT_FUNC
PyInit_myspkmeanssp(void){
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if(!m) return NULL;
    return m;
}



