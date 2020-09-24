#include<vector>
#include<CL/sycl.hpp>
namespace sycl = cl::sycl;
using namespace std;

#define SIZE 1024

int main()
{
    array<int, SIZE> a;
    for(int i = 0; i < SIZE; i++)
        a[i] = -1;
    sycl::range<1> a_size{SIZE};

    /*
	(A)
	The exception handler receives a list of std::exception_ptr . These are opaque, so a common
	approach is to rethrow it to then catch the actual exception. 
	This one terminates the process immediately.
    */

    auto exception_handler = [](sycl::exception_list exceptions){
        for(exception_ptr const& e : exceptions){
            try{
                rethrow_exception(e);
            }
            catch(sycl::exception const& e){
                cout << "ASYNCHRONOUS SYCL exception:\n" << e.what() << endl;
                terminate();
            }
        }
    };

    sycl::default_selector device_selector;

    // (B)
    // pass the exception handler to the queue constructor.

    sycl::queue d_queue(device_selector, exception_handler);

    sycl::buffer<int, 1> a_device(a.data(), a_size);

    d_queue.submit([&](sycl::handler &cgh){
        //we intentionally introduce an error here.  By not passing the Command Group Handler
	    //to the accessor constructor, we make a "host only" accessor which will 
	    //fail if used in the kernel
	    // uncomment 'cgh' argument to make this example run without error.
        auto a_w = a_device.get_access<sycl::access::mode::write>(/*cgh*/);

        cgh.parallel_for<class ex1>(a_size, [=](sycl::id<1> idx){
            a_w[idx] = 1;
        });
    });

    cout << "about to wait and throw" << endl;
    d_queue.wait_and_throw();

    //report out
    cout << "everything should be 1" << endl;
    for (int i = 0; i < 10; i++) {
	    cout << i << ": " << a[i] << endl;
    }
    for (int i = SIZE - 10; i < SIZE; i++) {
	    cout << i << ": " << a[i] << endl;
    }
	
	cout << "all done" << endl;
    return 0;
}