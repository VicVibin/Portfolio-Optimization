#include <iostream>
#include <string>
#include <chrono>

__device__ double power_d(double x, const int &i)
{
    if(i == 0){return 1;}
    if(i == 1){return x;}
    if(i < 0){return 1 / power_d(x, -i);}

    if(i % 2 == 0){int p = i / 2; return power_d(x, p) * power_d(x, p);}
    else
    {
        return x * power_d(x, i - 1);
    } 

}

__device__ double power_cube(double x, const int & i)
{
    if(i == 0){return 1;}
    if(i == 1){return x;}
    if(i < 0){return 1 / power_d(x, -i);}

    if(i % 2 == 0)
    {
        int p = i / 2; 
        return power_d(x, p) * power_d(x, p);
    }

    if( i % 3 == 0)
    {
        int p = i / 3;
        return power_d(x, p) * power_d(x, p) * power_d(x, p);
    }

    if(i % 5 == 0)
    {
        int p = i / 5;
        return power_d(x, p) * power_d(x, p) * power_d(x, p) * power_d(x, p) * power_d(x, p);
    }

    else
    {
        return x * power_d(x, i - 1);
    } 

}

__global__ void TS(double *b, const double S0, const double up, const double down, const int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < size + 1)
    {
        b[idx] = S0 * power_cube(down, size - idx) * power_cube(up, idx);
    }   
};

__global__ void update(double *a, const double q, const double disc, const int size)
{   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size + 1)
    {
        a[idx] = disc * (q * a[idx + 1]) + (1 - q) * a[idx]; 
    }
}

__global__ void max_call(double *b, const double K, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx  < size + 1)
    {double val = b[idx] - K;
    if (val < 0) {b[idx] = 0;}
    else {b[idx] = val;}}
}

__global__ void max_put(double *b, const double K, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx  < size + 1)
    {
    double val = K - b[idx];
    if (val < 0){b[idx] = 0;}
    else{b[idx] = val;}
    }
}

__global__ void Early_call(double *b, double *a, const double K, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx == size)
    {double val = b[idx] - K;
    if (val > a[idx])
    {a[idx] = val;}}
}

__global__ void Early_Put(double *b, double *a, const double K, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx  == size)
    {double val = K - b[idx];
    if (val > a[idx]) 
    {a[idx] = val;}}
}


void print(double *c, int size)
{
    for(int i = 0; i < size; i++)
    {
        std::cout << c[i] << "\t";
    }
    std::cout << "\n " << "__________ \n";
}

double power(double x, const int &i)
{
    if(i == 0 || x == 1){return 1;}
    if(i == 1 || x == 0){return x;}
    if(i < 0){return 1 / power(x, -i);}

    if(i % 2 == 0){int p = i / 2; return power(x, p) * power(x, p);}
    else{return x * power(x, i - 1);} 

}

double sqrt(double x)
{
    if (x <= 0)
    {
        return 0;
    }

    double t = 1;
    for(int i  = 0; i < 30; i++){t = 0.5 * (t + x / t);}
    return t;
}

double factorial(double n)
{
    if (n <= 1){return 1;}
    else
    {
        return n * factorial(n - 1);
    }
}

double exp(double x)
{   
    if (abs(x) < 1e-10){return 1;}

    int val = abs(x) / 2;
    int approx = 5 *(val + 1);
    double sum = 0;
    for(int i = 0; i < approx; i++)
    {
        sum += power(x, i) / factorial(i);
    }
    return sum;

}

double JR_Method_CPU(const double &S0, double &K, double &T, const double &sigma, const double &r, const int &step, std::string type = "C")
{   
    double zero = 0;
    double dt = T / step;
    double nu = r - 0.5 * power(sigma, 2);
    double up = exp(nu * dt + sigma * sqrt(dt));
    double down = 1 / up;
    double q = (exp(r * dt) -  down)/(up - down);
    double disc = exp(-r * dt);
    double *S = (double*)malloc((step + 1) * sizeof(double));
    double *C = (double*)malloc((step + 1) * sizeof(double));

    if (type == "C")
    {
        for(int i = 0; i < step + 1; i++)
        {   
            S[i] = S0 * power(down, step - i) * power(up, i);   
            C[i] = max(S[i] - K, zero);
        }
    }

    else
    {
        for(int i = 0; i < step + 1; i++)
        {
        S[i] = S0 * power(down, step - i) * power(up, i);   
        C[i] = max(K - S[i], zero);
        }   
    }


    for(int i = step - 1; i > -1; i--)
    {   
        for(int j = 0; j < i+1; j++)
        {   
            C[j] = disc * (q * C[j + 1]) + (1 - q) * C[j]; 
        }

        if(type == "C")
        {
            C[i] = max(S[i] - K, C[i]);
        }
        else
        {
            C[i] = max(K - S[i], C[i]);
        }

    }
    free(S);
    double value = C[0];
    free(C);
    return value;
}

double JR_Method_GPU(const double &S0, double &K, double &T, const double &sigma, const double &r, const int &step, std::string type = "C")
{   


    double dt = T / step;
    double disc = exp(-r * dt);
    double nu = r - 0.5 * power(sigma, 2);
    double up = exp(nu * dt + sigma * sqrt(dt));
    double down = 1 / up;
    double q = ((1/ disc) -  down)/(up - down);

    double *S, *C;
    
    cudaMalloc((void **)&S, (step + 1) * sizeof(double));
    cudaMalloc((void **)&C, (step + 1) * sizeof(double));

    int threadsPerBlock = 256;
    int blocksPerGrid = (step + threadsPerBlock) / threadsPerBlock;

    TS<<<blocksPerGrid, threadsPerBlock>>>(S, S0, up, down, step);
    cudaDeviceSynchronize();  
    cudaMemcpy(C, S, (step + 1) * sizeof(double), cudaMemcpyDeviceToDevice); // Copies data from S to C


    if (type == "C")
    {
        max_call<<<blocksPerGrid, threadsPerBlock>>>(C, K, step);

    }
    else
    {
        max_put<<<blocksPerGrid, threadsPerBlock>>>(C, K, step);
    }


    for(int i = step - 1; i > -1; i--)
    {   
        update<<<blocksPerGrid, threadsPerBlock>>>(C, q, disc, i);

        if(type == "C")
        {   
            Early_call<<<blocksPerGrid, threadsPerBlock>>>(S, C, K, i);
        }
        else
        {
            Early_Put<<<blocksPerGrid, threadsPerBlock>>>(S, C, K, i);
        }

        
    }   
        

    double *V = (double*)malloc(sizeof(double));
    cudaMemcpy(V, C, sizeof(double), cudaMemcpyDeviceToHost);
    double value = V[0];
    cudaFree(S);
    cudaFree(C);
    free(V);
    return value;
}


int main()
{
    double S0 = 251;
    double K = 251;
    double T = 0.3;
    double sigma = 0.53;
    double r = 0.03;
    int step = 1000;
    std::string type = "C";
    std::cout << "Option value with parameters:\n"
          << "  S0    = " << S0 << "\n"
          << "  K     = " << K << "\n"
          << "  T     = " << T << "\n"
          << "  sigma = " << sigma << "\n"
          << "  r     = " << r << "\n"
          << "  steps = " << step << "\n"
          << "  type  = " << type << "\n";

    auto start1 = std::chrono::high_resolution_clock::now();
    double algo = JR_Method_CPU(S0, K, T, sigma, r, step, type);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end1 - start1;
    std::cout << "Completed CPU computation, running GPU"<< "\n";
    auto start = std::chrono::high_resolution_clock::now();
    double option_price = JR_Method_GPU(S0, K, T, sigma, r, step, type);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Option Price GPU: $" << option_price << "\n";
    // std::cout << "Option Price CPU: $" << algo << "\n";
    std::cout << "Time elapsed=> CPU vs GPU for " << step << " steps: \n" 
    << elapsed1.count() <<"s" << " vs " << elapsed.count() << "s \n";
    return 0;
}