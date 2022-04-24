import myspkmeanssp as spkmeans
import numpy as np
import pandas as pd
import sys
       
def print_indices(ind):
    for i in range(len(ind)):
        ind[i] = (str)(ind[i])
    print(','.join(ind))  

def print_results(arr):
    for row in arr:
        for i in range(len(row)):
            row[i] = format(row[i],".4f")
            if row[i]=="-0.0000":
                row[i] = "0.0000"
        print(','.join(row))  

def kmeans_pp(df, N, d, k):
    np.random.seed(0)
    C = np.zeros((k,d)) # initialize centroids
    index = np.random.choice(N) # generate rand index_0.
    C_indices = [index]  # C_indices[0] = index_0
    C[0]=np.array(df[index]) # C[0] = df[index_0] = µ0
    # µi :=  df[index_i]
    i=1
    while(i<k):
        D = np.zeros(N) # D[l] = min{(xl − µj)^2}, ∀j 1≤j≤i
        P = np.zeros(N) # P[l] = D[l]/sum(D)
        for l in range(N): 
            D[l]= np.min([np.linalg.norm(df[l]-C[j])**2 for j in range(i)])
        for l in range(N): 
            P[l] = D[l]/np.sum(D)
        index = np.random.choice(N, p=P) # generate rand index_i according to P
        C_indices.append(index) # C_indices[i]= index_i
        C[i]=np.array([df[index]]) # C[i] = µi, ∀i 0≤i<k 
        i+=1  

    try:
        final_centroids = spkmeans.fit(df, C.tolist(), "kmeans", d, k)
    except:
        raise Exception("An Error Has Occurred")
        
    print_indices(C_indices)
    print_results(final_centroids)

############################# spkmeans #############################

try:
    k = int(sys.argv[1]) # number of clusters
    goal = sys.argv[2]
    file_name = sys.argv[3] 
except:
    raise Exception("Invalid Input!")

if goal not in ["wam", "ddg", "lnorm", "spk", "jacobi"]:
    raise Exception("Invalid Input!")

try:
    df = pd.read_csv(file_name, header=None)
except:
    raise Exception("Invalid Input!") 

N = df.shape[0] 
d = df.shape[1] 
df = df.to_numpy().tolist() 

if k>=N or k<0:
    raise Exception("Invalid Input!") 

try:
    final_matrix = spkmeans.fit(df, None, goal, d, k) # exe spkmeans on the data
except:
    raise Exception("An Error Has Occurred")

if(goal=="spk"):
    k = len(final_matrix[0])
    kmeans_pp(final_matrix, N, k, k)
else:
    print_results(final_matrix)
  
  



    
