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
#include "arguments.hpp"
#include "dictionary.hpp"
#include "initializer.hpp"
#include "vector.hpp"
#include "matrix.hpp"

#define LEFT_SAMPLING 0
#define RIGHT_SAMPLING 1
#define BOTH_SAMPLING 2
#define SAMPLING_STRATEGY 1

namespace poincare_disc{

  constexpr float EPS = 1e-6;

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Poincare Disc
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  RealType arcosh(const RealType x)
  {
    assert( x >= 1 );
    return std::log(x + std::sqrt(x*x - 1)); 
  }

  template <class RealType>
  struct Distance
  {
  public:
    using real = RealType;
  public:
    Distance(): u_(), v_(), uu_(), vv_(), uv_(), alpha_(), beta_(), gamma_() {}
    real operator()(const Vector<real>& u, const Vector<real>& v)
    {
      u_ = u;
      v_ = v;
      uu_ = u_.squared_sum();
      vv_ = v_.squared_sum();
      uv_ = u_.dot(v_);
      alpha_ = 1 - uu_;
      if(alpha_ <= 0){ alpha_ = EPS; } // TODO: ensure 0 <= uu_ <= 1-EPS;
      // if(!(alpha_ > 0)){ std::cout << "uu_: " << uu_ << ", alpha_: " << alpha_ << std::endl; }
      // assert(alpha_ > 0);
      beta_ = 1 - vv_;
      if(beta_ <= 0){ beta_ = EPS; } // TODO: ensure 0 <= vv_ <= 1-EPS;
      // if(!(beta_ > 0)){ std::cout << "vv_: " << vv_ << ", beta_: " << beta_ << std::endl; }
      // assert(beta_ > 0);
      gamma_ = 1 + 2 * (uu_ - 2 * uv_ + vv_) / alpha_ / beta_;
      if(gamma_ < 1.){ gamma_ = 1.; } // for nemerical error
      assert(gamma_ >= 1);
      return arcosh<real>(gamma_);
    }

    void backward(Vector<real>& grad_u, Vector<real>& grad_v, real grad_output)
    {
      real c = grad_output;
      if(gamma_ == 1){
        grad_u.zero_();
        grad_v.zero_();
        return;
      }

      c  *= 4 / std::sqrt(gamma_ * gamma_ - 1) / alpha_ / beta_;

      // grad for u
      real cu = c * alpha_ * alpha_ / 4;
      real cv = c * beta_ * beta_  / 4;

      grad_u.assign_(cu * (vv_ - 2 * uv_ + 1) / alpha_, u_);
      grad_u.add_(-cu, v_);

      grad_v.assign_(cv * (uu_ - 2 * uv_ + 1) / beta_, v_);
      grad_v.add_(-cv, u_);
    }

  private:
    Vector<real> u_, v_;
    real uu_, vv_, uv_, alpha_, beta_, gamma_;
  };

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Negative Sampler
  ///////////////////////////////////////////////////////////////////////////////////////////

  // TODO: reject pairs which appear in dataset
  struct UniformNegativeSampler
  {
  public:

    template <class InputIt>
    UniformNegativeSampler(InputIt first, InputIt last, unsigned int seed)
      : engine_(seed), dist_(first, last)
    {}

    std::size_t operator()()
    { return dist_(engine_); }

  private:
    std::default_random_engine engine_;
    std::discrete_distribution<std::size_t> dist_;
  };

 

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Optimization
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  struct LinearLearningRate
  {
  public:
    using real = RealType;
  public:
    LinearLearningRate(const real lr_init, const real lr_final, const std::size_t total_iter)
      :lr_init_(lr_init), lr_final_(lr_final), current_iter_(0), total_iter_(total_iter)
    {}
  public:
    void update(){ ++current_iter_;}
    real operator()() const
    {
      real r = static_cast<real>(static_cast<double>(current_iter_) / total_iter_);
      assert( 0 <= r && r <= 1);
      return (1-r) * lr_init_ + r * lr_final_;
    }
  public:
    real lr_init_;
    real lr_final_;
    std::size_t current_iter_;
    std::size_t total_iter_;
  };

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Poincare Embedding
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  struct Config
  {
    using real = RealType;
    std::size_t dim = 5; // dimension
    unsigned int seed = 0; // seed
    UniformInitializer<real> initializer = UniformInitializer<real>(-0.0001, 0.0001); // embedding initializer
    std::size_t num_threads = 1;
    std::size_t neg_size = 10;
    std::size_t max_epoch = 1;
    char delim = '\t';
    real lr0 = 0.01; // learning rate
    real lr1 = 0.0001; // learning rate
    std::size_t ws = 5;
  };

  template <class RealType>
  void clip(Vector<RealType>& v, const RealType& thresh = 1-EPS)
  {
    RealType vv = v.squared_sum();
    if(vv >= thresh*thresh){
      v.mult_(thresh / std::sqrt(vv));
    }
  }
  
  
  template <class RealType>
  bool train_line(const std::vector<std::int32_t>& line, 
                    std::vector<int32_t>& bow, 
                    Matrix<RealType>& embeddings, 
                    Vector<RealType>& u,
                    Vector<RealType>& v,
                    const Config& config, 
                    double& avg_loss, 
                    double&cum_loss, 
                    size_t thread_no) {
      
    for (int32_t w = 0; w < line.size(); w++) {
        u = embeddings[line[w]];
        Vector<RealType> v;
        // generate bow
        for (int32_t c = -config.ws; c <= config.ws; c++) {
          if (c != 0 && w + c >= 0 && w + c < line.size()) {
              v.add(1.0, embeddings[line[w+c])];
          }
        }
        v.mult(1.0 / (2*config.ws))
                
        embeddings
               
      exp_neg_dist_values[0] = std::exp(-dists[0](embeddings[i], embeddings[j]));
      for(std::size_t k = 0; k < config.neg_size; ++k){
        auto i = left_indices[k + 1] = negative_sampler();
        auto j = right_indices[k + 1] = itr->second;
        exp_neg_dist_values[k + 1] = std::exp(-dists[k + 1](embeddings[i], embeddings[j]));
      }

      // compute gradient
      // grads for 1, 2, ...
      // at first, compute the grad input
      real Z = exp_neg_dist_values[0];
      for(std::size_t k = 0; k < config.neg_size; ++k){
        Z += exp_neg_dist_values[k + 1];
      }
      for(std::size_t k = 0; k < config.neg_size; ++k){
        dists[k + 1].backward(left_grads[k+1], right_grads[k+1], -exp_neg_dist_values[k+1]/Z);
      }
      // grads for 0
      dists[0].backward(left_grads[0], right_grads[0], 1 - exp_neg_dist_values[0]/Z);

      // add loss
      {
        avg_loss -= std::log(exp_neg_dist_values[0]);
        avg_loss += std::log(Z);
      }

      // update
      for(std::size_t k = 0; k < 1 + config.neg_size; ++k){
        auto i = left_indices[k], j = right_indices[k];
        embeddings[i].add_clip_(-lr(), left_grads[k]);
        embeddings[j].add_clip_(-lr(), right_grads[k]);
      }

      lr.update();

      // next iteration
      ++itr;
  }
  

  template <class RealType>
  bool train_thread(const std::string& infile,
                    Matrix<RealType>& embeddings,
                    Dictionary<std::string>& dict, 
                    const Config<RealType>& config,
                    LinearLearningRate<RealType>& lr,
                    const std::size_t thread_no,
                    const std::size_t token_count_per_thread,
                    const unsigned int seed)
  {
    using real = RealType;
    
    // clip
    for(std::size_t i = 0, I = embeddings.nrow(); i < I; ++i){
      clip(embeddings[i]);
    }

    // construct negative sampler
    UniformNegativeSampler negative_sampler(counts.begin(), counts.end(), seed);

    // data, gradients, distances
    std::vector<std::size_t> left_indices(1 + config.neg_size), right_indices(1 + config.neg_size);
    Matrix<real> left_grads(1 + config.neg_size, config.dim, ZeroInitializer<real>()); // u
    Matrix<real> right_grads(1 + config.neg_size, config.dim, ZeroInitializer<real>()); // v, v', ...
    std::vector<Distance<real>> dists(1 + config.neg_size);
    std::vector<real> exp_neg_dist_values(1 + config.neg_size);
    
    // start training
    auto tick = std::chrono::system_clock::now();
    auto start_time = tick;
    constexpr std::size_t progress_interval = 10000;
    double avg_loss = 0;
    double cum_loss = 0;
    
    std::ifstream ifs(infile);
    utils::seek(ifs, threadId * utils::size(ifs) / config.num_threads);
    int64_t localTokenCount = 0;
    std::vector<int32_t> line;
    std::vector<int32_t> bow;
    
    while (localTokenCount < token_count_per_thread) {
        if(thread_no == 0 && localTokenCount % progress_interval == 0){
            auto tack = std::chrono::system_clock::now();
            auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(tack-tick).count();
            tick = tack;
            double percent = (100.0 * localTokenCount) / token_count_per_thread;
            cum_loss += avg_loss;
            avg_loss /= progress_interval;
            std::cout << "\r"
                      <<std::setw(5) << std::fixed << std::setprecision(2) << percent << " %"
                      << "    " << config.num_threads * progress_interval*1000./millisec << " itr/sec"
                      << "    " << "loss: " << avg_loss
                      << std::flush;

            avg_loss = 0;
        }
          
        dict.getLine(ifs, line);
        
        localTokenCount += train_line(line, bow, dists, exp_neg_dist_values, config, avg_loss, cum_loss, thread_no); 
    }
    
    ifs.close();

    if(thread_no == 0){
      cum_loss += avg_loss;
      auto tack = std::chrono::system_clock::now();
      auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(tack-start_time).count();
        std::cout << "\r"
                  <<std::setw(5) << std::fixed << std::setprecision(2) << 100 << " %"
                  << "    " << config.num_threads * total_itr * 1000./millisec << " itr/sec"
                  << "    " << "loss: " << cum_loss / total_itr
                  << std::endl;
    }
    return true;
  }

  template <class RealType>
  bool poincare_embedding(const std::string& infile, 
                          Matrix<RealType>& embeddings,
                          const Dictionary<std::string>& dict,
                          const Config<RealType>& config)
  {
    using real = RealType;

    std::default_random_engine engine(config.seed);

    embeddings.init(dict.size(), config.dim, config.initializer);

    std::cout << "embedding size: " << embeddings.nrow() << " x " << embeddings.ncol() << std::endl;

    // fit
    LinearLearningRate<real> lr(config.lr0, config.lr1, dict.tokenCount() * config.max_epoch);
    std::vector<std::pair<std::size_t, std::size_t> > fake_pairs(config.neg_size);
    std::cout << "num_threads = " << config.num_threads << std::endl;
    std::size_t data_size_per_thread = dict.tokenCount() / config.num_threads;
    std::cout << "data size = " << data_size_per_thread << "/thread" << std::endl;

    for(std::size_t epoch = 0; epoch < config.max_epoch; ++epoch){
      std::cout << "epoch " << epoch+1 << "/" << config.max_epoch << " start" << std::endl;

      if(config.num_threads > 1){
        // multi thread

        std::vector<std::thread> threads;
        for(std::size_t i = 0; i < config.num_threads; ++i){
          unsigned int thread_seed = engine();
          threads.push_back(std::thread( [=, &embeddings, &counts, &lr]{ train_thread(infile, embeddings, dict,                                                                                      config, lr, i, data_size_per_thread, thread_seed); }  ));
        }
        for(auto& th : threads){
          th.join();
        }
      }else{
        // single thread
        const unsigned int thread_seed = engine();
        train_thread(embeddings, dict.counts(), data.begin(), data.end(), config, lr, 0, thread_seed);
      }

    }

    return true;
  }

}

#endif
