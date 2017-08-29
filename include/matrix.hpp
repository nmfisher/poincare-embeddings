template <class RealType>
  struct Matrix
  {
  public:
    using real = RealType;
  public:
    Matrix(): m_(0), n_(0), rows_() {}

    template <class Initializer>
    Matrix(const std::size_t m, const std::size_t n, Initializer initializer): m_(), n_(), rows_()
    { init(m, n, initializer); }

    Matrix(const Matrix<real>& mat): m_(mat.m_), n_(mat.n_), rows_(mat.rows_) {}

    Matrix<real>& operator=(const Matrix<real>& mat)
    {
      m_ = mat.m_;
      n_ = mat.n_;
      rows_ = mat.rows_;
      return *this;
    }

  public:

    template <class Initializer>
    void init(const std::size_t m, const std::size_t n, Initializer initializer)
    {
      m_ = m; n_ = n; rows_ = std::vector<Vector<real> >(m);
      for(std::size_t i = 0; i < m; ++i){
        rows_[i] = Vector<real>(std::shared_ptr<real>(new real[n]), n);
        for(std::size_t j = 0; j < n; ++j){
          rows_[i][j] = initializer();
        }
      }
    }

    std::size_t nrow() const { return m_; }
    std::size_t ncol() const { return n_; }

    const Vector<real>& operator[](const std::size_t i) const
    { return rows_[i]; }

    Vector<real>& operator[](const std::size_t i)
    { return rows_[i]; }

    Matrix<real>& zero_()
    {
      for(std::size_t i = 0; i < m_; ++i){
        rows_[i].zero_();
      }
      return *this;
    }

  private:
    std::size_t m_, n_;
    std::vector<Vector<real> > rows_;
  };
