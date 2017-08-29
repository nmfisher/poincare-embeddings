///////////////////////////////////////////////////////////////////////////////////////////
  // Vector, Matrix
  ///////////////////////////////////////////////////////////////////////////////////////////

  template <class RealType>
  struct Vector
  {
  public:
    using real = RealType;
  public:
    Vector(): data_(nullptr), dim_(0) {}
    Vector(std::shared_ptr<real> data, std::size_t dim): data_(data), dim_(dim) {}
    Vector(const Vector<real>& v): data_(v.data_), dim_(v.dim_) {}
    Vector<real>& operator=(const Vector<real>& v)
    {
      data_ = v.data_; dim_ = v.dim_;
      return *this;
    }
  public:
    const std::size_t dim() const { return dim_; }
    const real operator[](const std::size_t i) const { return data_.get()[i]; }
    real& operator[](const std::size_t i) { return data_.get()[i]; }

    Vector<real>& assign_(const real c, const Vector<real>& v)
    {
      if(dim_ != v.dim_){
        dim_ = v.dim_;
        data_ = std::shared_ptr<real>(new real[v.dim_]);
      }
      for(int i = 0, I = dim(); i < I; ++i){
        data_.get()[i] = c * v.data_.get()[i];
      }
      return *this;
    }

    Vector<real>& zero_()
    {
      for(int i = 0, I = dim(); i < I; ++i){
        data_.get()[i] = 0;
      }
      return *this;
    }

    Vector<real>& add_(const real c, const Vector<real>& v)
    {
      for(int i = 0, I = dim(); i < I; ++i){
        data_.get()[i] += c * v.data_.get()[i];
      }
      return *this;
    }

    Vector<real>& add_clip_(const real c, const Vector<real>& v, const real thresh=1.0-EPS)
    {
      real uu = this->squared_sum(), uv = this->dot(v), vv = v.squared_sum();
      real C = uu + 2*c*uv + c*c*vv; // resulting norm
      real scale = 1.0;
      if(C > thresh * thresh){
        scale = thresh / sqrt(C);
      }
      assert( 0 < scale && scale <= 1. );
      if(scale == 1.){
        for(int i = 0, I = dim(); i < I; ++i){
          data_.get()[i] += c * v.data_.get()[i];
        }
      }else{
        for(int i = 0, I = dim(); i < I; ++i){
          data_.get()[i] = (data_.get()[i] + c * v.data_.get()[i]) * scale;
        }
      }
      assert(this->squared_sum() <= (thresh + EPS) * (thresh+EPS));
      return *this;
    }

    Vector<real>& mult_(const real c)
    {
      for(int i = 0, I = dim(); i < I; ++i){
        data_.get()[i] *= c;
      }
      return *this;
    }

    real squared_sum() const { return this->dot(*this); }

    real dot(const Vector& v) const
    { return std::inner_product(data_.get(), data_.get() + dim_, v.data_.get(), 0.); }


  private:
    std::size_t dim_;
    std::shared_ptr<real> data_;
  };


  template <class RealType>
  std::ostream& operator<<(std::ostream& out, const Vector<RealType>& v)
  {
    if(v.dim() < 5){
      out << "[";
      for(int i = 0; i < v.dim(); ++i){
        if(i > 0){ out << ", ";}
        out << v[i];
      }
      out << "]";
    }else{
      out << "[";
      out << v[0] << ", " << v[1] << ", ..., " << v[v.dim()-1];
      out << "]";
    }
    return out;
  }