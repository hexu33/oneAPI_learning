#include<array>
#include<CL/sycl.hpp>
using namespace std;
namespace sycl = cl::sycl;

const int SIZE = 64;

int main()
{
    array<int,SIZE> a, c;
    array<sycl::float4, SIZE> b;
    for(int i = 0; i < SIZE; i++)
    {
        a[i] = i;
        b[i] = (float)-i;
        c[i] = i;
    }

    {
        sycl::range<1> a_size{SIZE};
        
        sycl::queue d_queue;

        sycl::buffer<int> a_device(a.data(), a_size);
        sycl::buffer<int> c_device(c.data(), a_size);
        sycl::image<2> b_device(b.data(), sycl::image_channel_order::rgba,
            sycl::image_channel_type::fp32, sycl::range<2>(8, 8));

        d_queue.submit([&](sycl::handler &cgh){
            sycl::accessor<int, 1, sycl::access::mode::discard_write , sycl::access::target::global_buffer>   c_res(c_device, cgh);
            sycl::accessor<int, 1, sycl::access::mode::read          , sycl::access::target::constant_buffer> a_res(a_device, cgh);
            sycl::accessor<sycl::float4, 2, sycl::access::mode::write, sycl::access::target::image> b_res(b_device, cgh);

            sycl::float4 init = {0.f, 0.f, 0.f, 0.f};

            cgh.parallel_for<class ex1>(a_size, [=](sycl::id<1> idx){
                c_res[idx] = a_res[idx];
                b_res.write(sycl::int2(0, 0), init);
            });
        });
    }
    return 0;
}