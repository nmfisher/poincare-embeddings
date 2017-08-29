///////////////////////////////////////////////////////////////////////////////////////////
  // Initializer
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  struct Initializer
  {
  public:
    using real = RealType;
  public:
    virtual real operator()() = 0;
  };

  template <class RealType>
  struct ZeroInitializer: public Initializer<RealType>
  {
  public:
    using real = RealType;
  public:
    real operator()() { return 0.; }
  };

  template <class RealType>
  struct UniformInitializer: public Initializer<RealType>
  {
  public:
    using real = RealType;
  private:
    std::default_random_engine engine_;
    std::uniform_real_distribution<real> dist_;
  public:
    UniformInitializer(const real a, const real b, const unsigned int seed = 0)
      : engine_(seed), dist_(a, b)
    {}
  public:
    real operator()() { return dist_(engine_); }
  };