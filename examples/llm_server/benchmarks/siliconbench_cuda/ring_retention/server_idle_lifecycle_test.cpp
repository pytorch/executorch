// Diagnostic tests against the real Runner/ServingRuntime with a fake executor.
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/runner.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>
#include <executorch/extension/llm/serving/serving_runtime.h>
#include <executorch/runtime/platform/runtime.h>
#include <pytorch/tokenizers/tokenizer.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace b=executorch::extension::llm::batching;
namespace s=executorch::extension::llm::serving;
namespace llm=executorch::extension::llm;
using namespace std::chrono_literals;

void require(bool value,const char* message) {
  if (!value) throw std::runtime_error(message);
}
template<class P> void wait_until(P ready) {
  const auto deadline=std::chrono::steady_clock::now()+3s;
  while (!ready()) {
    require(std::chrono::steady_clock::now()<deadline,"bounded event timeout");
    std::this_thread::sleep_for(1ms);
  }
}
struct Gate {
  std::mutex mutex;
  std::condition_variable cv;
  bool entered=false, released=false;
  void arrive() {
    std::unique_lock<std::mutex> lock(mutex);entered=true;cv.notify_all();
    require(cv.wait_for(lock,3s,[&]{return released;}),"gate release timeout");
  }
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    require(cv.wait_for(lock,3s,[&]{return entered;}),"gate entry timeout");
  }
  void release() {std::lock_guard<std::mutex> lock(mutex);released=true;cv.notify_all();}
};
struct Tokenizer final : tokenizers::Tokenizer {
  tokenizers::Error load(const std::string&) override {return tokenizers::Error::Ok;}
  tokenizers::Result<std::vector<uint64_t>> encode(const std::string&,int8_t,int8_t) const override {return std::vector<uint64_t>{4,5};}
  tokenizers::Result<std::string> decode(uint64_t,uint64_t token,bool) const override {return std::to_string(token);}
  tokenizers::Result<std::string> id_to_piece(uint64_t token) const override {return std::to_string(token);}
  tokenizers::Result<uint64_t> piece_to_id(const std::string&) const override {return uint64_t{4};}
};
b::GenConfig generation(int count) {b::GenConfig c;c.max_new_tokens=count;return c;}
s::PromptInput prompt() {return s::PromptInput{{llm::make_token_input({4,5})}};}
s::RequestHandle submit(s::ServingRuntime& runtime,std::function<void(s::GenerationEvent)> callback={}) {
  s::GenerationOptions options;options.max_new_tokens=8;
  auto r=runtime.generate(std::nullopt,prompt(),options,std::move(callback));
  require(std::holds_alternative<s::RequestHandle>(r),"serving request rejected");
  return std::get<s::RequestHandle>(std::move(r));
}

int main() {
  try {
    executorch::runtime::runtime_init();
    nlohmann::json passed=nlohmann::json::array();
    {
      b::testing::FakeExecutor executor;std::atomic<int> calls{0},outputs{0};
      b::IdleMaintenanceConfig idle;idle.delay=60ms;
      idle.callback=[&]{require(!executor.in_execute(),"idle concurrent execute");++calls;return true;};
      b::Runner runner(executor,b::DecodeFirstScheduler::create(64,4,32),idle);
      auto owner=runner.open_session_async().get();require(owner.has_value(),"owner");
      std::this_thread::sleep_for(140ms);require(calls==0,"startup maintenance");
      executor.hold();
      auto h=owner->generate_async(std::vector<b::Token>{4,5},generation(8),[&](auto& u){outputs+=u.tokens.size();});
      wait_until([&]{return executor.in_execute();});std::this_thread::sleep_for(140ms);
      require(calls==0,"maintenance during prefill/decode");executor.release();h.wait();
      wait_until([&]{return calls==1;});std::this_thread::sleep_for(140ms);require(calls==1,"repeated idle maintenance");
      auto next=owner->generate_async(std::vector<b::Token>{6},generation(8),[&](auto& u){outputs+=u.tokens.size();});next.wait();
      wait_until([&]{return calls==2;});runner.shutdown();
      require(outputs==16,"changed generated output count");
      passed.push_back("idle only after work; no maintenance during forward; once per idle epoch; 16 outputs");
    }
    {
      b::testing::FakeExecutor executor;std::atomic<int> attempts{0},success{0};std::atomic<bool> ready{false};
      b::IdleMaintenanceConfig idle;idle.delay=40ms;
      idle.callback=[&]{++attempts;if(!ready)return false;++success;return true;};
      b::Runner runner(executor,b::DecodeFirstScheduler::create(64,4,32),idle);
      auto owner=runner.open_session_async().get();auto h=owner->generate_async(std::vector<b::Token>{4,5},generation(1),{});h.wait();
      wait_until([&]{return attempts>=2;});require(success==0,"ignored deferred readiness");ready=true;
      wait_until([&]{return success==1;});auto stable=attempts.load();std::this_thread::sleep_for(140ms);
      require(attempts==stable,"successful maintenance retries");runner.shutdown();
      passed.push_back("outer-runtime deferral and one successful idle attempt");
    }
    {
      b::testing::FakeExecutor executor;Gate hook;std::atomic<int> calls{0};
      b::IdleMaintenanceConfig idle;idle.delay=40ms;idle.callback=[&]{if(++calls==1)hook.arrive();return true;};
      b::Runner runner(executor,b::DecodeFirstScheduler::create(64,4,32),idle);
      auto owner=runner.open_session_async().get();auto h=owner->generate_async(std::vector<b::Token>{4},generation(1),{});h.wait();hook.wait();
      auto next=owner->generate_async(std::vector<b::Token>{5},generation(8),{});
      require(!next.done(),"new request was not queued during held maintenance");hook.release();next.wait();
      require(next.finish_reason()==b::FinishReason::NewTokenLimit,"lost queued request");runner.shutdown();
      passed.push_back("arrival during maintenance queues without runner-lock deadlock");
    }
    for(int mode=0;mode<3;++mode) {
      b::testing::FakeExecutor executor;Tokenizer tokenizer;Gate gate;std::atomic<int> calls{0};
      s::ServingRuntimeConfig config;config.max_sessions=4;config.max_requests=4;config.max_context_length=128;
      config.idle_maintenance_delay=60ms;config.idle_maintenance=[&]{++calls;};
      s::ServingRuntime runtime(executor,b::DecodeFirstScheduler::create(64,4,32),tokenizer,config);
      s::RequestHandle first;
      if(mode==0) {
        first=submit(runtime,[&](s::GenerationEvent e){if(std::holds_alternative<s::TerminalEvent>(e))gate.arrive();});
        gate.wait();std::this_thread::sleep_for(140ms);require(calls==0,"idle during terminal delivery");gate.release();first.wait();
      } else if(mode==1) {
        first=submit(runtime);first.wait();
        s::GenerationOptions options;options.max_new_tokens=8;
        auto result=runtime.generate(std::nullopt,[&](const s::PromptPreparationContext&)->s::PromptPreparationResult{gate.arrive();return prompt();},options,{});
        require(std::holds_alternative<s::RequestHandle>(result),"deferred request rejected");
        auto second=std::get<s::RequestHandle>(std::move(result));gate.wait();std::this_thread::sleep_for(140ms);
        require(calls==0,"idle while pending request preparation");gate.release();second.wait();
      } else {
        first=submit(runtime);first.wait();
        runtime.open_session_async("lifecycle",[&](s::LifecycleResult r){require(!r,"lifecycle failed");gate.arrive();});
        gate.wait();std::this_thread::sleep_for(140ms);require(calls==0,"idle during lifecycle callback");gate.release();
      }
      wait_until([&]{return calls==1;});std::this_thread::sleep_for(140ms);require(calls==1,"repeated serving idle hook");runtime.shutdown();
      passed.push_back(mode==0?"terminal callback fence":mode==1?"pending preparation fence":"lifecycle callback and capture destruction fence");
    }
    std::cout<<nlohmann::json{{"passed",passed},{"cases",passed.size()},{"fake_executor",true},{"GPU_model_tested",false}}.dump()<<'\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
