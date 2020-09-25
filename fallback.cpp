// The following code fails due to the size of the workgroup when executed on Intel Processor Graphics, such as Intel HD Graphics 530.
// The SYCL specification allows specifying a secondary queue as a parameter to the submit function
// and this secondary queue is used if the device kernel runs into issues with submission to the first device.

#include<CL/sycl.hpp>
#include<iostream>
using namespace cl::sycl;

constexpr int N = 1024;
constexpr int M = 32;

int main()
{
    cpu_selector CpuSelector;
    queue cpuQueue(CpuSelector);
    queue defaultqueue;
    buffer<int, 2> buf(range<2>(N, N));
    defaultqueue.submit([&](handler &cgh){
        auto bufacc = buf.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class ndim>(nd_range<2>(range<2>(N, N), range<2>(M, M)), [=](nd_item<2> i){
            id<2> ind = i.get_global_id();
            bufacc[ind[0]][ind[1]] = ind[0] + ind[1];    
        });
    }, cpuQueue);
    auto bufacc1 = buf.get_access<access::mode::read>();
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            if(bufacc1[i][j] != i + j)
            {
                std::cout << "Wrong result.\n";
                return 1;
            }
    std::cout << "Correct result.\n";
    return 0;
}