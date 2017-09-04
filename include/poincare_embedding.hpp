#ifndef POINCARE_EMBEDDING_HPP
#define POINCARE_EMBEDDING_HPP

#include <cassert>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <iomanip>
#include "config.hpp"
#include "dictionary.hpp"

constexpr float EPS = 1e-6;

#include "vector.hpp"
#include "matrix.hpp"
#include "utils.hpp"
#include "model.hpp"

namespace poincare_disc { 

   
  ///////////////////////////////////////////////////////////////////////////////////////////
  // Poincare Embedding
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  template <class RealType>
  void clip(Vector<RealType>& v, const RealType& thresh = 1-EPS)
  {
    RealType vv = v.squared_sum();
    if(vv >= thresh*thresh){
      v.mult_(thresh / std::sqrt(vv));
    }
  }
    
  template <class RealType>
  bool train_thread(const std::string& infile,
                    Matrix<RealType>& w_i,
                    Matrix<RealType>& w_o,
                    Dictionary<std::string>& dict, 
                    const Config<RealType>& config,
                    LinearLearningRate<RealType>& lr,
                    const std::size_t thread_no,
                    const std::size_t token_count_per_thread,
                    const unsigned int seed,
                    const size_t epoch,
                    UniformNegativeSampler& negative_sampler)
  {
        
    // clip
    for(std::size_t i = 0, I = w_i.nrow(); i < I; ++i){
      clip(w_i[i]);
      clip(w_o[i]);
    }
    
    // start training
    auto tick = std::chrono::system_clock::now();
    auto start_time = tick;
    constexpr std::size_t progress_interval = 10000;
    
    double avg_loss = 0;
    double cum_loss = 0;
    
    std::ifstream ifs(infile);
        
    size_t start = thread_no * size(ifs) / config.num_threads;
    size_t end = (thread_no + 1) * size(ifs) / config.num_threads;
    seek(ifs, start);
    int64_t localTokenCount = 0;
    int64_t totalTokenCount = 0;
    std::vector<int32_t> line;
    std::vector<int32_t> target; 
    std::vector<int32_t> window;
    
    Model<RealType> model(w_i, w_o, negative_sampler, lr, config, ZeroInitializer<RealType>());
        
    while (ifs.tellg() < end) {        
    
        dict.getLine(ifs, line);
                
        for (int32_t w = 0; w < line.size(); w++) {
            
            localTokenCount++;
            if(thread_no == 0 && localTokenCount % progress_interval == 0 && ifs.tellg() < end){
                auto tack = std::chrono::system_clock::now();
                auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(tack-tick).count();
                tick = tack;
                double percent;
                percent = 100.0 * ifs.tellg() / end;
                cum_loss += avg_loss;
                std::cout << "\r"
                          <<std::setw(5) << std::fixed << std::setprecision(5) << percent << " %"
                          << "    " << localTokenCount*1000./millisec << " tokens/sec/thread"
                          << "    " << "loss: " << avg_loss / progress_interval
                          << "    " << "lr: " << lr()
                          << "    " << "epoch: " << epoch << " / " << config.max_epoch 
                          << std::flush;
                avg_loss = 0;
                totalTokenCount += localTokenCount;
                localTokenCount = 0;
            }
                        
            target.push_back(line[w]);
            
            // generate context window
            for (int32_t c = -config.ws; c <= config.ws; c++) {
              if (c != 0 && w + c >= 0 && w + c < line.size()) {
                  window.push_back(line[w+c]);
              }
            }
            
            if(window.size() == 0) {
                continue;
            }
            if(config.model == ModelType::CBOW) {
                avg_loss += model.update(window, target);    
            } else if (config.model == ModelType::SKIPGRAM) {
                avg_loss += model.update(target, window);
            } else {
                std::cout << config.model;
            }
            
            window.clear();
            target.clear();
        }
    }
    
    if(thread_no == 0){
        auto tack = std::chrono::system_clock::now();
        auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(tack-tick).count();
        tick = tack;
        double percent = 100.0;
        cum_loss += avg_loss;
        std::cout << "\r"
                  <<std::setw(5) << std::fixed << std::setprecision(5) << percent << " %"
                  << "    " << localTokenCount *1000./millisec << " tokens/sec/thread"
                  << "    " << "loss: " << cum_loss / totalTokenCount
                  << "    " << "lr: " << lr()
                  << "    " << "epoch: " << epoch << " / " << config.max_epoch 
                  << std::flush;

        avg_loss = 0;
    }    
    ifs.close();

    return true;
  }

  template <class RealType>
  bool poincare_embedding(const std::string& infile, 
                          Matrix<RealType>& w_i,
                          Matrix<RealType>& w_o,
                          Dictionary<std::string>& dict,
                          const Config<RealType>& config)
  {
    using real = RealType;

    std::default_random_engine engine(config.seed);

    w_i.init(dict.size(), config.dim, config.initializer);
    w_o.init(dict.size(), config.dim, config.initializer);
    
    std::cout << "embedding size: " << w_i.nrow() << " x " << w_i.ncol() << std::endl;

    // fit
    LinearLearningRate<real> lr(config.lr0, config.lr1, dict.tokenCount() * config.max_epoch);
    std::cout << "num_threads = " << config.num_threads << std::endl;
    
    std::size_t data_size_per_thread = (size_t) dict.tokenCount() / config.num_threads;
    std::cout << "data size = " << data_size_per_thread << "/thread" << std::endl;

    for(std::size_t epoch = 0; epoch < config.max_epoch; ++epoch){
        
        std::cout << std::endl;
            
      const unsigned int thread_seed = engine();
      
      // construct negative sampler, shared between threads
      std::vector<size_t> counts = dict.counts();
      UniformNegativeSampler negative_sampler(counts.begin(), counts.end(), thread_seed);

        // multi thread
        if(config.num_threads > 1){
            std::vector<std::thread> threads;
            for(std::size_t i = 0; i < config.num_threads; ++i){    
              threads.push_back(std::thread( [&infile, &w_i, &w_o, &dict, &config, &lr, i, data_size_per_thread, thread_seed, epoch, &negative_sampler ]() mutable { train_thread(infile, w_i, w_o, dict, config, lr, i, data_size_per_thread, thread_seed, epoch, negative_sampler); }  ));
            }
            for(auto& th : threads){
              th.join();
            }
        // single thread
        } else{
            train_thread(infile, w_i, w_o, dict, config, lr, 0, data_size_per_thread, thread_seed, epoch, negative_sampler);
        }
    }

    return true;
  }

}

#endif
