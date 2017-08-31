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

constexpr float EPS = 1e-6;

#include "vector.hpp"
#include "matrix.hpp"
#include "utils.hpp"

#define LEFT_SAMPLING 0
#define RIGHT_SAMPLING 1
#define BOTH_SAMPLING 2
#define SAMPLING_STRATEGY 1

namespace poincare_disc { 


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
      
      beta_ = 1 - vv_;
      if(beta_ <= 0){ beta_ = EPS; } // TODO: ensure 0 <= vv_ <= 1-EPS;
      
      gamma_ = 1 + 2 * (uu_ - 2 * uv_ + vv_) / alpha_ / beta_;
      
      if(gamma_ < 1.){ gamma_ = 1.; } // for nemerical error
      assert(gamma_ >= 1);
      return arcosh<real>(gamma_);
    }

    
    // partial Euclidean derivative of dist(u,v) wrt u = (4  / (beta * sqrt(gamma^2 - 1))) * (((||v||^2 - 2<u,v> + 1)/alpha^2)))*u - v/alpha)
    // partial Euclidean derivative of dist(u,v) wrt v is the same with alphas/betas swapped
    // Riemannian derivative is (A * Euclidean derivative),  where A is ((1 - ||x||^2)^2 / 4), the inverse of the Riemannian metric tensor)
    void backward(Vector<real>& grad_u, Vector<real>& grad_v, real log_loss_grad)
    {
      real c = log_loss_grad;
      
      if(gamma_ == 1){
        grad_u.zero_();
        grad_v.zero_();
        return;
      }

      c *= 1 / std::sqrt(gamma_ * gamma_ - 1) / beta_ / alpha_;
      real cu = c / alpha_; 
      real cv = c / beta_;
  
      grad_u.assign_(cu * (vv_ - 2 * uv_ + 1), u_);
      grad_u.add_(-cu, v_);

      grad_v.assign_(cv * (uu_ - 2 * uv_ + 1) / beta_ / beta_, v_);
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
    std::size_t dim; // dimension
    unsigned int seed = 0; // seed
    UniformInitializer<real> initializer = UniformInitializer<real>(-0.0001, 0.0001); // embedding initializer
    std::size_t num_threads = 1;
    std::size_t neg_size;
    std::size_t max_epoch;
    char delim = '\t';
    real lr0 = 0.01; // learning rate
    real lr1 = 0.0001; // learning rate
    int32_t ws;
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
  bool train_thread(const std::string& infile,
                    Matrix<RealType>& w_u,
                    Matrix<RealType>& w_v,
                    Dictionary<std::string>& dict, 
                    const Config<RealType>& config,
                    LinearLearningRate<RealType>& lr,
                    const std::size_t thread_no,
                    const std::size_t token_count_per_thread,
                    const unsigned int seed,
                    const size_t epoch)
  {
    using real = RealType;
        
    // clip
    // for(std::size_t i = 0, I = w_u.nrow(); i < I; ++i){
      // clip(w_u[i]);
      // clip(w_v[i]);
    // }


    // construct negative sampler
    std::vector<size_t> counts = dict.counts();
    UniformNegativeSampler negative_sampler(counts.begin(), counts.end(), seed);

    // data, gradients, distances
    std::vector<std::size_t> left_indices(1 + config.neg_size), right_indices(1 + config.neg_size);
    Matrix<real> grads_u(1 + config.neg_size, config.dim, ZeroInitializer<real>()); 
    Matrix<real> grads_v(1 + config.neg_size, config.dim, ZeroInitializer<real>()); 
    
    // vector to hold the target word embeddings
    Vector<real> u(config.dim);
    
    // vector to hold the average of the context embeddings 
    Vector<real> v(config.dim);
    
    // vector to hold the Poincare ball distances
    std::vector<Distance<real>> dists(1 + config.neg_size);
    
    // negative sampling loss 
    RealType loss;
    
    // start training
    auto tick = std::chrono::system_clock::now();
    auto start_time = tick;
    constexpr std::size_t progress_interval = 1000;
    
    double avg_loss = 0;
    double cum_loss = 0;
    
    std::ifstream ifs(infile);
        
    size_t start = thread_no * size(ifs) / config.num_threads;
    size_t end = (thread_no + 1) * size(ifs) / config.num_threads;
    seek(ifs, start);
    int64_t localTokenCount = 0;
    std::vector<int32_t> line;
    std::vector<int32_t> bow;
    std::string token;
        
    while (ifs.tellg() < end) {        
    
        dict.getLine(ifs, line);
                
        for (int32_t w = 0; w < line.size(); w++) {
            
            localTokenCount++;
            if(thread_no == 0 && localTokenCount % progress_interval == 0){
                auto tack = std::chrono::system_clock::now();
                auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(tack-tick).count();
                tick = tack;
                double percent = 100.0 * localTokenCount / token_count_per_thread;
                cum_loss += avg_loss;
                std::cout << "\r"
                          <<std::setw(5) << std::fixed << std::setprecision(5) << percent << " %"
                          << "    " << config.num_threads * progress_interval*1000./millisec << " tokens/sec/thread"
                          << "    " << "loss: " << avg_loss / progress_interval
                          << "    " << "lr: " << lr()
                          << "    " << "epoch: " << epoch << " / " << config.max_epoch 
                          << std::flush;

                avg_loss = 0;
            }
            
            loss = 0;
            
            u.assign_(1.0, w_u[line[w]]);
            v.zero_();
            
            bow.clear();
            
            // generate bow
            for (int32_t c = -config.ws; c <= config.ws; c++) {
              if (c != 0 && w + c >= 0 && w + c < line.size()) {
                  v.add_(1.0, w_v[line[w+c]]);
                  bow.push_back(line[w+c]);
              }
            }
            
            if(bow.size() == 0) {
                continue;
            }
            
            v.mult_(1.0 / bow.size());
                                                            
            // calculate distances / sigmoids / log loss
            loss = std::log(1 / (1 + std::exp(-dists[0](u, v))));
                                    
            for(std::size_t k = 1; k < config.neg_size; ++k){
                loss += 1 / (1 + std::exp(-dists[k](u, w_v[negative_sampler()])));
            }

            
            loss /= config.neg_size;
                                
            // calculate grads
            for(std::size_t k = 0; k < config.neg_size; ++k){
                grads_u[k].zero_();
                grads_v[k].zero_();
                dists[k].backward(grads_u[k], grads_v[k], loss);
            }
    
            // update
            for(std::size_t k = 0; k < config.neg_size; ++k){
                w_u[line[w]].add_clip_(-lr(), grads_u[k]);
                for(std::size_t j = 0; j < bow.size(); ++j) {
                    w_v[bow[j]].add_clip_(-lr(), grads_v[k]); 
                }
            }            
            
            lr.update();
            avg_loss += loss;
            
        }
    }
    
    if(thread_no == 0){
            auto tack = std::chrono::system_clock::now();
            auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(tack-tick).count();
            tick = tack;
            double percent = 100.0 * localTokenCount / token_count_per_thread;
            cum_loss += avg_loss;
            std::cout << "\r"
                      <<std::setw(5) << std::fixed << std::setprecision(5) << percent << " %"
                      << "    " << config.num_threads * progress_interval*1000./millisec << " tokens/sec/thread"
                      << "    " << "loss: " << avg_loss / progress_interval
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
                          Matrix<RealType>& w_u,
                          Matrix<RealType>& w_v,
                          Dictionary<std::string>& dict,
                          const Config<RealType>& config)
  {
    using real = RealType;

    std::default_random_engine engine(config.seed);

    w_u.init(dict.size(), config.dim, config.initializer);
    w_v.init(dict.size(), config.dim, config.initializer);
    
    std::cout << "embedding size: " << w_u.nrow() << " x " << w_u.ncol() << std::endl;

    // fit
    LinearLearningRate<real> lr(config.lr0, config.lr1, dict.tokenCount() * config.max_epoch);
    std::cout << "num_threads = " << config.num_threads << std::endl;
    
    std::size_t data_size_per_thread = (size_t) dict.tokenCount() / config.num_threads;
    std::cout << "data size = " << data_size_per_thread << "/thread" << std::endl;

    for(std::size_t epoch = 0; epoch < config.max_epoch; ++epoch){
        
        std::cout << std::endl;
            
      const unsigned int thread_seed = engine();

        // multi thread
        if(config.num_threads > 1){
            std::vector<std::thread> threads;
            for(std::size_t i = 0; i < config.num_threads; ++i){    
              threads.push_back(std::thread( [&infile, &w_u, &w_v, &dict, &config, &lr, i, data_size_per_thread, thread_seed, epoch ]() mutable { train_thread(infile, w_u, w_v, dict, config, lr, i, data_size_per_thread, thread_seed, epoch); }  ));
            }
            for(auto& th : threads){
              th.join();
            }
        // single thread
        } else{
            train_thread(infile, w_u, w_v, dict, config, lr, 0, data_size_per_thread, thread_seed, epoch);
        }
    }

    return true;
  }

}

#endif