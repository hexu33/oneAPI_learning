#include<iostream>
#include<array>
#include<CL/sycl.hpp>

using namespace std;
namespace sycl = cl::sycl;

constexpr int arr_size = 1024;

template<typename T>
class Vassign{
    T val_;
    sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer> access_;

    public:
    Vassign(T val, sycl::accessor<Y, 1, sycl::accessor::modfe::read_write, sycl::access::target::global_buffer>& access)
    : val_(val), access_(access){}

    void operator()(sycl::id<1> idx) {access_[idx] = idx[0] + val_;}
};

void UseSingleTask(array<int, arr_size>& a){
    sycl::range<1> a_size{a.size()};
    sycl::buffer<int, 1> a_buf(a.data(), a_size);
    sycl::queue q;

    q.submit([&](sycl::handler& h){
        auto a_in = a_buf.get_access<sycl::access::mode::write>(h);
        h.single_task([=]() {a_in[0] = -1;})
    });
}

void UseParallelFor(array<int, arr_size>& a){
    sycl::range<1> a_size{a.size()};
    sycl::buffer<int, 1> a_buf(a.data(), a_size);
    sycl::queue q;

    q.submit([&](sycl::handler& h){
        auto a_in = a_buf.get_access<sycl::access::mode::read_write, sycl::access::target::global_buffer>(h);
        Vassign<int> F(10000, a_in);
        h.parallel_for(sycl::range<1>(arr_size), F);
    });
}

int main()
{
    array<int, arr_size> a;
    for(int i = 0; i < arr_size; i++)
    {
        a[i] = i;
    }
    UseSingleTask(a);
    cout << "Expected -1: " << a[0] << "\n";

    UseParallelFor(a);
    cout << "Expected 10000 + i : " << a[0] << ", " << a[1] << ", ... , " << a[arr_size - 1] << "\n";
    return 0;
}