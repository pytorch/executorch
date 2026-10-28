// Diagnostic snapshots only; no launch arguments or arithmetic are changed.
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
namespace {
bool full_history_captured = false;
int capture_ordinal = 0, pending = -1, pending_tokens = 0, pending_oh = 0, pending_ot = 0, pending_dimension = 256;
const char* directory() { return std::getenv("ET_DIAGNOSTIC_ATTENTION_OUTPUT"); }
void checked(cudaError_t e) { if (e != cudaSuccess) std::abort(); }
int integer(void** a, int i) { return *static_cast<int*>(a[i]); }
char* pointer(void** a, int i) { return *static_cast<char**>(a[i]); }
void write(const std::string& name, const char* data, size_t n) {
  std::ofstream f(std::string(directory()) + "/" + name, std::ios::binary);
  f.write(data, n); if (!f.good()) std::abort();
}
void packed(const std::string& name, const char* data, int tokens, int heads, int hs, int ts, int dimension = 256) {
  const size_t n = (heads-1)*size_t(hs)+(tokens-1)*size_t(ts)+dimension;
  if (hs<1 || ts<1 || n>size_t(tokens)*heads*dimension*2) std::abort();
  std::vector<char> raw(n*2), out(size_t(tokens)*heads*dimension*2);
  checked(cudaMemcpy(raw.data(),data,raw.size(),cudaMemcpyDeviceToHost));
  for(int t=0;t<tokens;++t) for(int h=0;h<heads;++h)
    std::memcpy(out.data()+(size_t(t)*heads+h)*dimension*2,raw.data()+(size_t(h)*hs+size_t(t)*ts)*2,dimension*2);
  write(name,out.data(),out.size());
}
}
extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction fn,unsigned gx,unsigned gy,unsigned gz,
 unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** args,void** extra) {
 using Launch=CUresult(CUDAAPI*)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**);
 static auto real=reinterpret_cast<Launch>(dlsym(RTLD_NEXT,"cuLaunchKernel"));if(!real)std::abort();
 const char* name=nullptr;
 bool direct_output=false;
 if(directory() && args && cuFuncGetName(&name,fn)==CUDA_SUCCESS && name) {
  bool ring=std::strcmp(name,"_decode_reference_order")==0;
  bool base=std::strcmp(name,"_sdpa_varlen_decode_kernel")==0;
  bool generic=std::strcmp(name,"_sdpa_varlen_kernel")==0;
  unsigned tensor_tokens=gx;
  if(generic){int t=integer(args,9);generic=t>4 && t<=64 && gy==8 && gz==1;if(generic)tensor_tokens=unsigned(t);}
  bool full=base && gy==1 && (gz==4 || gz==8 || gz==16 || gz==32 || gz==64 || gz==128);
  if((ring||base||generic) && (gy==8 || full)) {
   if(tensor_tokens<1 || tensor_tokens>64 || (!full && gz!=12 && !generic) || pending!=-1)std::abort();
   checked(cudaStreamSynchronize(stream));
   int h=ring?14:8; if(integer(args,h)!=16)std::abort();
   // Exported T=1 is constexpr; T>1 is an i32 launch argument.
   int delta=generic?1:(tensor_tokens==1?0:1); if(delta && integer(args,h+1)!=int(tensor_tokens))std::abort();
   direct_output=generic;
   int id=capture_ordinal++;pending=id;pending_tokens=tensor_tokens;pending_dimension=full?512:256;
   pending_oh=integer(args,(ring?23:17)+delta);pending_ot=integer(args,(ring?24:18)+delta);
   auto prefix=std::to_string(id)+"-";
   packed(prefix+"Q.bf16",pointer(args,0),tensor_tokens,16,integer(args,h+1+delta),integer(args,h+2+delta),pending_dimension);
   std::vector<int> info(162*7);checked(cudaMemcpy(info.data(),pointer(args,ring?13:7),info.size()*4,cudaMemcpyDeviceToHost));
   write(prefix+"info.i32",reinterpret_cast<char*>(info.data()),info.size()*4);
   const bool capture_history=full && !full_history_captured;
   for(int kind=0;kind<2;++kind) {
    if(ring) packed(prefix+(kind?"V.bf16":"K.bf16"),pointer(args,kind?4:3),tensor_tokens,8,integer(args,kind?7:5),integer(args,kind?8:6));
    else {
     const int kv_dimension=full?512:2048;
     std::vector<char> out(size_t(tensor_tokens)*kv_dimension*2);
     std::vector<char> history(capture_history?size_t(tensor_tokens)*1025*512*2:0);int cols=info[4];if(cols<1 || cols>82)std::abort();
     for(unsigned t=0;t<tensor_tokens;++t) {
      int found=-1;for(int s=0;s<162;++s)if(t>=unsigned(info[s*7]) && t<unsigned(info[s*7]+info[s*7+1]) && info[s*7+1]>0){if(found!=-1)std::abort();found=s;}
      if(found<0)std::abort();int s=found*7;if(info[s+1]!=1 || info[s+3]<1025 || info[s+3]>1090)std::abort();
      int col=info[s+2];if(col<0 || col>=cols)std::abort();int row=(info[s+3]-1)%3071;
      int cs=integer(args,(kind?15:12)+delta),hs=integer(args,(kind?16:13)+delta);if(cs!=kv_dimension || hs!=(full?512:256))std::abort();
      size_t address_row=full?size_t(info[s+5]+info[s+3]-1):(size_t(row)*cols+col);
      const char* p=pointer(args,kind?2:1)+address_row*cs*2;
      checked(cudaMemcpy(out.data()+size_t(t)*kv_dimension*2,p,kv_dimension*2,cudaMemcpyDeviceToHost));
      if(capture_history) {
       if(info[s+3]!=1025)std::abort();
       const char* first=pointer(args,kind?2:1)+size_t(info[s+5])*cs*2;
       checked(cudaMemcpy(history.data()+size_t(t)*1025*512*2,first,1025*512*2,cudaMemcpyDeviceToHost));
      }
     }
     write(prefix+(kind?"V.bf16":"K.bf16"),out.data(),out.size());
     if(capture_history)write(prefix+(kind?"V-history.bf16":"K-history.bf16"),history.data(),history.size());
    }
   }
   if(capture_history)full_history_captured=true;
   std::ofstream m(std::string(directory())+"/"+prefix+"metadata.json");
   m<<"{\"kernel\":\""<<name<<"\",\"ordinal\":"<<id<<",\"tokens\":"<<tensor_tokens<<",\"logical_end\":"<<info[3]<<",\"columns\":"<<info[4]<<",\"head_dimension\":"<<pending_dimension<<"}";
  }
 }
 CUresult result=real(fn,gx,gy,gz,bx,by,bz,shared,stream,args,extra);
 if(result==CUDA_SUCCESS && directory() && name && pending>=0 &&
  (direct_output || std::strcmp(name,"_reduce")==0 || std::strcmp(name,"_sdpa_varlen_reduce_kernel")==0)) {
  checked(cudaStreamSynchronize(stream));packed(std::to_string(pending)+"-attention-out.bf16",pointer(args,3),pending_tokens,16,pending_oh,pending_ot,pending_dimension);pending=-1;
 }
 return result;
}
