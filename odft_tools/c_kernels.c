static void eval_rbf_kernel(double* K, 
                            unsigned int n, 
                            unsigned int m, 
                            unsigned int n_dim) {
    //printf("Ordered tons of spam!\n");
    int i, j;
    int nn_dim = n*(1 + n_dim);
    int mn_dim = m*(1 + n_dim);
    for (i = 0; i < nn_dim; i++) {
        for (j = 0; j < mn_dim; j++) {
            K[i * mn_dim + j] = 1.0/n_dim;
        }
    }
}


static void eval_rbf_kernel_omp(double* K, 
                                unsigned int n, 
                                unsigned int m, 
                                unsigned int n_dim) {
    //printf("Ordered tons of spam!\n");
    int i, j;
    int nn_dim = n*(1 + n_dim);
    int mn_dim = m*(1 + n_dim);
    #pragma omp parallel
    {
        #pragma omp for
        for (i = 0; i < nn_dim; i++) {
            for (j = 0; j < mn_dim; j++) {
                //#pragma omp atomic write
                K[i * mn_dim + j] = 1.0/n_dim;
            }
        }
    }
}