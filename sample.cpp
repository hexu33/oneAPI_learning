#include<CL/sycl.hpp>
#include<array>
using namespace std;
namespace sycl = cl::sycl;

constexpr int arr_size = 1024;

int main()
{
    array<int, arr_size> a_arr, b_arr, c_arr;

    for(int i = 0; i < arr_size; i++)
        a_arr[i] = b_arr[i] = c_arr[i] = i;
    
    auto platforms = sycl::platform::get_platform();

    for(auto &platform: platforms)
    {
        cout << "Platform:" << platform.get_info<sycl::info::platform::name>() << endl;
        auto devices = platform.get.devices();
        for(auto &device: devices)
            cout << "Device:" << device.get_info<sycl::info::device::name>() < endl;
    }
    sycl::default_selector device_selector;
    sycl::queue q(device_selector);

    sycl::range<1> a_size(arr_size);

    sycl::buffer<int, 1> a_buf(a_arr.data(), a_arr.size());
    sycl::buffer<int, 1> b_buf(b_arr.data(), b_arr.size());
    sycl::buffer<int, 1> c_buf(c_arr.data(), c_arr.size());

    auto e = q.submit([&](sycl::handler &h)
    {
        auto c = c_buf.get_access<sycl::access::mode::write>(h);
        auto b = b_buf.get_access<sycl::access::mode::read>(h);
        auto a = a_buf.get_access<sycl::access::mode::read>(h);

        h.parallel_for(a_size, [=](sycl::id<1> idx){c[idx] = a[idx] + b[idx];});
    });
    e.wait();

    auto c = c_buf.get_access<sycl::access::mode::read>();
    for(int i = 0; i < arr_size; i++)
        cout << c[i] << " ";
    cout << endl;

    return 0;
}