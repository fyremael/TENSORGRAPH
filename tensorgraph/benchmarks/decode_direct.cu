// Independent handwritten attribution baseline. Never labeled generated TENSORGRAPH.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>

template<typename T> __global__ void direct(const T* x,const T* r,const T* w,T* a,T* b,int h) {
    using Reduce=cub::BlockReduce<float,256>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float inverse;
    float accum=0.f;
    for(int j=threadIdx.x;j<h;j+=blockDim.x){
        int i=blockIdx.x*h+j;
        float s=float(x[i])+float(r[i]);
        accum+=s*s;
    }
    float sum=Reduce(temp).Sum(accum);
    if(threadIdx.x==0)inverse=rsqrtf(sum/float(h)+1e-5f);
    __syncthreads();
    for(int j=threadIdx.x;j<h;j+=blockDim.x){
        int i=blockIdx.x*h+j;
        float s=float(x[i])+float(r[i]);
        a[i]=T(s);b[i]=T((s*inverse)*float(w[j]));
    }
}
void direct_out(torch::Tensor x,torch::Tensor r,torch::Tensor w,torch::Tensor a,torch::Tensor b){
    auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,x.scalar_type(),"direct_decode",[&]{
        direct<scalar_t><<<x.size(0),256,0,stream.stream()>>>(x.data_ptr<scalar_t>(),r.data_ptr<scalar_t>(),w.data_ptr<scalar_t>(),a.data_ptr<scalar_t>(),b.data_ptr<scalar_t>(),x.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
