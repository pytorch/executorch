// Real CUDA mini inference through ServingRuntime; forced output lengths.
#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/serving/serving_runtime.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/tokenizer.h>
#include <gflags/gflags.h>
#include <nlohmann/json.hpp>
#if SB_SERVER_IDLE_HOOK
#include "idle_shared_workset.h"
#endif
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <thread>

DEFINE_string(pte,"","Original mini PTE");
DEFINE_string(data_path,"","Identical original PTD");
DEFINE_int32(concurrency,1,"Named sessions in each request group");
DEFINE_int32(budget,4096,"Original exported bulk token budget");
DEFINE_bool(shared_idle_refresh,false,"Enable optional idle shared-pool refresh");
namespace cb=executorch::backends::cuda::batching;
namespace b=executorch::extension::llm::batching;
namespace s=executorch::extension::llm::serving;
namespace llm=executorch::extension::llm;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
int64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
struct Tokenizer final:tokenizers::Tokenizer {
  tokenizers::Error load(const std::string&) override{return tokenizers::Error::Ok;}
  tokenizers::Result<std::vector<uint64_t>> encode(const std::string&,int8_t,int8_t) const override{return std::vector<uint64_t>{4,5};}
  tokenizers::Result<std::string> decode(uint64_t,uint64_t token,bool) const override{return std::string("t")+std::to_string(token);}
  tokenizers::Result<std::string> id_to_piece(uint64_t token) const override{return std::to_string(token);}
  tokenizers::Result<uint64_t> piece_to_id(const std::string&) const override{return uint64_t{4};}
};
Json memory(){
  require(cudaDeviceSynchronize()==cudaSuccess,"memory snapshot fence");
  size_t free=0,total=0;require(cudaMemGetInfo(&free,&total)==cudaSuccess,"memory info");
  int device=0;require(cudaGetDevice(&device)==cudaSuccess,"device");
  Json r={{"device_used_bytes",total-free}};
  auto pool=executorch::extension::cuda::CudaAllocator::pool_for_device(device);
  if(pool){uint64_t used=0,reserved=0,high=0;
    require(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemCurrent,&used)==cudaSuccess,"used");
    require(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved)==cudaSuccess,"reserved");
    require(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemHigh,&high)==cudaSuccess,"high");
    r["shared_used_bytes"]=used;r["shared_reserved_bytes"]=reserved;r["shared_used_high_bytes"]=high;
  }
  return r;
}
struct Response {std::mutex mutex;int64_t first_text_ns=0;std::optional<s::TerminalEvent> terminal;};
int main(int argc,char** argv){
  gflags::ParseCommandLineFlags(&argc,&argv,true);
  const int output=dup(STDOUT_FILENO);require(output>=0,"stdout");
  require(dup2(STDERR_FILENO,STDOUT_FILENO)>=0,"diagnostic stdout");
  FILE* protocol=fdopen(output,"w");
  try{
    executorch::runtime::runtime_init();
    require(FLAGS_concurrency==1||FLAGS_concurrency==16,"concurrency");
#if !SB_SERVER_IDLE_HOOK
    require(!FLAGS_shared_idle_refresh,"Old runtime cannot enable new hook");
#endif
    cb::CudaExecutorConfig config;config.model_path=FLAGS_pte;config.data_files.push_back(FLAGS_data_path);
    config.max_sessions=21;config.max_session_tokens=131072;config.initial_capacity=512;
    config.options.cuda_graph_for_static_methods=true;
    auto created=cb::CudaExecutor::create(config);require(created.ok(),"create executor");auto executor=std::move(*created);
    Tokenizer tokenizer;Json idle_events=Json::array();std::mutex idle_mutex;std::atomic<int> idle_count{0};
    s::ServingRuntimeConfig serving;serving.max_sessions=16;serving.max_context_length=131072;
    serving.max_requests=16;serving.max_pending_operations=64;serving.max_events_per_request=256;serving.max_tokens_per_request=256;
#if SB_SERVER_IDLE_HOOK
    serving.idle_maintenance_delay=std::chrono::milliseconds(100);
    if(FLAGS_shared_idle_refresh)serving.idle_maintenance=[&]{
      const auto begin=now();auto refresh=siliconbench::refresh_idle_shared_workset();const auto end=now();
      {std::lock_guard<std::mutex> lock(idle_mutex);
       idle_events.push_back({{"begin_ns",begin},{"end_ns",end},{"used",refresh.used},{"used_high",refresh.used_high},
         {"reserved_before",refresh.reserved_before},{"reserved_after",refresh.reserved_after},{"keep",refresh.keep},{"cuda_error",int(refresh.error)}});}
      require(refresh.error==cudaSuccess,"idle pool maintenance");++idle_count;
    };
#endif
    s::ServingRuntime runtime(*executor,b::DecodeFirstScheduler::create(FLAGS_budget,16,FLAGS_budget-16),tokenizer,serving);
    const auto ready_deadline=Clock::now()+std::chrono::seconds(120);
    while(!runtime.info().ready){require(Clock::now()<ready_deadline,"runtime initialization timeout");std::this_thread::sleep_for(std::chrono::milliseconds(2));}
    for(int i=0;i<FLAGS_concurrency;++i)require(!runtime.open_session_async("owner"+std::to_string(i)).get(),"named owner open");
    Json result={{"new_runtime",bool(SB_SERVER_IDLE_HOOK)},{"idle_refresh_enabled",FLAGS_shared_idle_refresh},
      {"concurrency",FLAGS_concurrency},{"budget",FLAGS_budget},{"initial_memory",memory()},
      {"decoder_is_deterministic_diagnostic_text",true},{"native_serving_API",true},{"groups",Json::array()}};
    std::vector<std::vector<uint64_t>> histories(FLAGS_concurrency);
    for(int burst=0;burst<3;++burst){
      Json inputs=Json::array();
      for(int owner=0;owner<FLAGS_concurrency;++owner){
        for(int n=0;n<1024;++n)histories[owner].push_back(3+(burst*1024+n+owner*17)%18000);
        inputs.push_back(histories[owner]);
      }
      std::vector<std::shared_ptr<Response>> observations;std::vector<s::RequestHandle> handles;
      const auto begin=now();
      for(int owner=0;owner<FLAGS_concurrency;++owner){
        auto response=std::make_shared<Response>();observations.push_back(response);
        s::GenerationOptions opts;opts.max_new_tokens=64;opts.sampling.temperature=0;
        auto r=runtime.generate("owner"+std::to_string(owner),s::PromptInput{{llm::make_token_input(histories[owner])}},opts,
          [response](s::GenerationEvent event){std::lock_guard<std::mutex> lock(response->mutex);
            if(auto* text=std::get_if<s::TextEvent>(&event)){if(!text->text.empty()&&!response->first_text_ns)response->first_text_ns=now();}
            else if(auto* terminal=std::get_if<s::TerminalEvent>(&event))response->terminal=std::move(*terminal);
          });
        require(std::holds_alternative<s::RequestHandle>(r),"real serving admission");handles.push_back(std::get<s::RequestHandle>(std::move(r)));
      }
      for(auto& h:handles)h.wait();const auto end=now();
      Json outputs=Json::array(),responses=Json::array();
      for(int owner=0;owner<FLAGS_concurrency;++owner){
        auto& observed=*observations[owner];std::lock_guard<std::mutex> lock(observed.mutex);
        require(observed.terminal.has_value(),"terminal callback");auto& t=*observed.terminal;
        require(t.finish_reason==s::FinishReason::Length&&!t.error&&t.stats.generated_token_ids.has_value(),"real serving finish");
        const auto& ids=*t.stats.generated_token_ids;require(ids.size()==64&&t.stats.completion_tokens==64,"actual output length");
        histories[owner].insert(histories[owner].end(),ids.begin(),ids.end());outputs.push_back(ids);
        responses.push_back({{"first_text_ns",observed.first_text_ns},{"completion_tokens",t.stats.completion_tokens},
          {"prefilled_prompt_tokens",t.stats.prefilled_prompt_tokens},{"reused_prompt_tokens",t.stats.reused_prompt_tokens},
          {"native_prefill_ms",t.stats.prefill_ms},{"native_decode_ms",t.stats.decode_ms},{"native_total_ms",t.stats.total_ms}});
      }
      auto before=memory();std::this_thread::sleep_until(Clock::time_point(std::chrono::nanoseconds(end))+std::chrono::milliseconds(200));
      auto after=memory();require(idle_count==(FLAGS_shared_idle_refresh?burst+1:0),"automatic idle epoch count");
      result["groups"].push_back({{"burst",burst},{"begin_ns",begin},{"end_ns",end},{"group_wall_ms",double(end-begin)/1e6},
        {"new_input_tokens_each",1024},{"actual_output_tokens_each",64},{"input_tokens",inputs},{"output_tokens",outputs},
        {"requests",responses},{"before_idle",before},{"after_idle",after},{"idle_budget_ms",200}});
    }
    runtime.shutdown();result["after_shutdown_memory"]=memory();
    {std::lock_guard<std::mutex> lock(idle_mutex);result["idle_events"]=idle_events;}
    std::fprintf(protocol,"%s\n",result.dump().c_str());std::fflush(protocol);std::fclose(protocol);return 0;
  }catch(const std::exception& e){std::fprintf(protocol,"%s\n",Json{{"failed",true},{"detail",e.what()}}.dump().c_str());std::fflush(protocol);std::fclose(protocol);return 1;}
}
