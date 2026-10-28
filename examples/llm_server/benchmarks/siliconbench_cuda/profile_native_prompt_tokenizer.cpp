// CPU-only prompt encoding diagnostic; no model or CUDA execution.
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
int main(int argc,char** argv){
  if(argc!=3)throw std::runtime_error("tokenizer and rendered prompt required");
  tokenizers::HFTokenizer tokenizer;
  if(tokenizer.load(argv[1])!=tokenizers::Error::Ok)throw std::runtime_error("tokenizer load");
  std::ifstream input(argv[2]);if(!input)throw std::runtime_error("prompt read");
  std::string prompt((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
  nlohmann::json records=nlohmann::json::array();
  for(int round=1;round<=4;++round){
    auto begin=std::chrono::steady_clock::now();
    for(int iteration=0;iteration<100;++iteration){
      auto result=tokenizer.encode(prompt,0,0);
      if(!result.ok() || result->size()!=1066)throw std::runtime_error("actual prompt encoding length");
    }
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    records.push_back({{"round",round},{"iterations",100},{"total_seconds",elapsed},{"per_encode_ms",elapsed*10}});
  }
  std::cout<<nlohmann::json{{"records",records},{"tokens_without_numeric_BOS",1066},{"CPU_only",true}}.dump()<<"\n";
}
