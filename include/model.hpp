namespace poincare_disc { 

   ///////////////////////////////////////////////////////////////////////////////////////////
  // Poincare Disc Distance / Gradient Functions
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
    // Riemannian derivative is (A * Euclidean derivative),  where A is ((1 - ||x||^2)^2 / 4), the inverse of the Riemannian metric tensor) - not shown here
    void backward(Vector<real>& grad_u, Vector<real>& grad_v, real grad_output)
    {
      real c = grad_output;
      
      if(gamma_ == 1){
        grad_u.zero_();
        grad_v.zero_();
        return;
      }

      c *= 4 / std::sqrt(gamma_ * gamma_ - 1);
      real cu = c / beta_; 
      real cv = c / alpha_;
  
      grad_u.assign_(cu * (vv_ - 2 * uv_ + 1) / alpha_ / alpha_, u_);
      grad_u.add_(-cu / alpha_, v_);

      grad_v.assign_(cv * (uu_ - 2 * uv_ + 1) / beta_ / beta_, v_);
      grad_v.add_(-cv / beta_, u_);
      
    }

  private:
    Vector<real> u_, v_;
    real uu_, vv_, uv_, alpha_, beta_, gamma_;
  };

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Negative Sampler
  ///////////////////////////////////////////////////////////////////////////////////////////
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
  // Learning Rate Optimizer
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
  // Model
  ///////////////////////////////////////////////////////////////////////////////////////////
    template <class RealType> 
    class Model {
      private:
        Vector<RealType> u_; // vector to hold the average of the target word embeddings
        Vector<RealType> v_; // vector to hold the average of the context embeddings 
        std::vector<Distance<RealType>> dists_; // vector to hold the Poincare ball distances
        Matrix<RealType>& w_i_;
        Matrix<RealType>& w_o_;
        // data, gradients, distances
        LinearLearningRate<RealType>& lr_;
        const Config<RealType>& config_;
        UniformNegativeSampler& negative_sampler_;
        Matrix<RealType> grads_u_;
        Matrix<RealType> grads_v_;
        std::vector<RealType> exp_neg_dist_values_;
        
      public:
            // \param w_i the matrix of input embeddings
            // \param w_o the matrix of output embeddings
            // \param lr the learning rate to apply  
            // \param the model configuration
            Model(Matrix<RealType>& w_i, 
                    Matrix<RealType>& w_o,
                    UniformNegativeSampler& negative_sampler,
                    LinearLearningRate<RealType>& lr,
                    const Config<RealType>& config,
                    const ZeroInitializer<RealType>& zeroInit) : 
                        w_i_(w_i),
                        w_o_(w_o),
                        lr_(lr),
                        config_(config),
                        negative_sampler_(negative_sampler),
                        u_(config.dim), 
                        v_(config.dim), 
                        dists_(1 + config.neg_size),
                        exp_neg_dist_values_(1 + config.neg_size),
                        grads_u_(1 + config.neg_size, config.dim, zeroInit),
                        grads_v_(1 + config.neg_size, config.dim, zeroInit)
                        { }
            // \param input an array of token IDs for the input (context window in CBOW or target word in skipgram)
            // \param target an array of token IDs for the target
            RealType update(const std::vector<int32_t>& input, const std::vector<int32_t>& target) {
                    u_.zero_();
                    v_.zero_();
                    grads_u_.zero_();
                    grads_v_.zero_();
                    
                    // construct average input embeddings
                    for(auto it=input.begin(); it < input.end(); it++) {
                        u_.add_(1.0, w_i_[*it]);    
                    }
                    u_.mult_(1.0 / input.size());
                    
                    // construct average target embeddings
                    for(auto it=target.begin(); it < target.end(); it++) {
                        v_.add_(1.0, w_o_[*it]);
                    }
                                    
                    v_.mult_(1.0 / target.size());
                        
                    // calculate exponentiated negative distances
                    exp_neg_dist_values_[0] = std::exp(-dists_[0](u_, v_));
                    RealType Z = 0.0;
                    for(std::size_t k = 1; k < config_.neg_size+1; ++k){
                        exp_neg_dist_values_[k] = std::exp(-dists_[k](u_, w_o_[negative_sampler_()]));
                        Z += exp_neg_dist_values_[k];
                    }            
                                    
                    // calculate Euclidean gradient vector 
                    dists_[0].backward(grads_u_[0], grads_v_[0], 1.0);

                    for(std::size_t k = 1; k < config_.neg_size+1; ++k){
                        dists_[k].backward(grads_u_[k], grads_v_[k], 1.0);
                        grads_u_[0].add_(-exp_neg_dist_values_[k] / Z, grads_u_[k]);
                    }
                        
                    // calculate Riemannian gradient vector
                    grads_u_[0].mult_(std::pow(1 - u_.squared_sum(), 2.0) / 4);
                    grads_v_[0].mult_(std::pow(1 - v_.squared_sum(), 2.0) / 4);
                
                    // weight update for input matrix 
                    for(auto it=input.begin(); it < input.end(); it++) {
                        w_i_[*it].add_clip_(-lr_(), grads_u_[0]);
                    }
                        
                    // weight update for target matrix 
                    for(auto it=target.begin(); it < target.end(); it++) {
                        w_o_[*it].add_clip_(-lr_(), grads_v_[0]); 
                    }
                                    
                    lr_.update();
                    // std::cout << exp_neg_dist_values_[0] / Z << std::endl;
                    // explicitly calculate loss (display purposes only, not used in derivative)
                    return std::log(exp_neg_dist_values_[0] / Z);
                } 
      };
};