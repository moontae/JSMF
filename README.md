# JSMF
Joint Stochastic Matrix Factorization (Matlab)

Co-occurrence information is powerful statistics that can model various discrete objects by their joint instances with other objects. Transforming unsupervised problems of learning low-dimensional geometry into provable decompositions of co-occurrence information, spectral inference provides fast algorithms and optimality guarantees for non-linear dimensionality reduction and latent topic analysis. Spectral approaches reduce the dependence on the original training examples, thereby producing substantial gain in efficiency, but at costs:
  
 - The algorithms perform poorly on real data that does not necessarily follow underlying models.  
 - Users can no longer infer information about individual examples, which is often important for real-world applications.  
 - Model complexity rapidly grows as the number of objects increases, requiring a careful curation of the vocabulary.   

The first issue is called model-data mismatch, which is a fundamental problem common in every spectral inference method for latent variable models. As real data never follows any particular computational model, this issue must be addressed for practicality of the spectral inference beyond synthetic settings. 

For the second issue, users could revisit probabilistic inference to infer latent mixtures about individual examples, but this brings back all the drawbacks of traditional approaches. Thresholded Linear Inverse algorithm is recently developed for individual inference, but it works only on tiny models, quickly losing its performance for the datasets whose underlying structures exhibit realistic correlations. 

While probabilistic inference also suffers from the third issue, the problem is more serious for spectral inferences because co-occurrence information easily exceeds storable capacity as the size of vocabulary becomes larger. If one can effectively compress the co-occurrence information, the model can easily process millions of examples.

We cast the learning problem in the framework of Joint Stochastic Matrix Factorization (JSMF) that can tackle each of three problems. Though it is named matrix factorization, the proposed algorithm integrates probabilistic characteristics, combining the benefit of the both worlds.


# Usages

The package consists of three main functions.

1) Running "factorizeC" directly factorizes the full co-occurrnece C with various options.
  - For the 'rectifier', you can choose 'AP'(Alteranting Projection), 'DC'(Diagonal Completion), 'DP'(Dykstra Projection), or 'Baseline'(no rectification).
  - For the 'optimizer', you can choose 'activeSet'(Active Set Method), 'admmDR' (ADMM with Douglas-Rachford splitting), or 'expGrad' (Exponentiated Gradient).
  - 'dataset' is just an indicator for the name of dataset for creating a proper log file.

2) Running "factorizeC_viaY" first compresses+rectifies the full co-occurrence C into Y and call "factorizeY".
  - For the 'rectifier', you can choose 'ENN-trunEig'(truncated Eigendecomposition), 'ENN-randEig'(randomized Eigendecomposition), 'PALM'(Proximal Alternating Linearized Minimization), 'IPALM'(Intertial PALM), or 'Baseline'(no rectification).
  - Same to the above for the 'optimizer'.

3) Running "factorizeVD_viaY" rectifies the compressed co-occurrence (V, D) into Y and call "factorizeY".
  - For the 'rectifier', you can only choose 'ENN' or 'Baseline'(no rectification) because we do not have an exact gradient information from the full co-occurrence C.
  - Same to the above for the 'optimizer'.

For each of three executions, 'EXP_factorize' runs overall experiments and store the learned models with the evaluation results.


# Cluster

In order to run experiments on the magma cluster, 

1) Compile the matlab code by running '_make.m' under the proper subfolder of 'experiments' folder. Once compiled at Matlab R2018a, it generates both an executable and a shell script files with several additional files.

2) Run 'send2magma.sh' (e.g., ./send2magma.sh EXP_factorizeC netid). It removes the additionally generated files and send the executables with the relevant shell scripts to your local folder at the magma cluster.

3) In your local folder at the magma cluster, run 'execute_all.sh' (e.g., ./execute_all.sh EXP_factorizeC). It will use the learned models in the shared folder, but saving your experiments under your local folder (e.g., JSMF/codes/experiments/EXP_factorizeC/models)


# Visualization

R code of generating plots will be soon added.
