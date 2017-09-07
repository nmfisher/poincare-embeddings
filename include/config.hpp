#include "initializer.hpp"

typedef enum {CBOW=1, SKIPGRAM} ModelType;

#include "arguments.hpp"
  
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
    ModelType model;
  };