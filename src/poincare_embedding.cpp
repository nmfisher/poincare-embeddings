#include <iostream>
#include <string>
#include <fstream>
#include "poincare_embedding.hpp"

using namespace poincare_disc;
using real = double;

void save(const std::string& filename,
          const Matrix<real>& embeddings,
          const Dictionary<std::string>& dict)
{
  std::ofstream fout(filename.c_str());
  if(!fout || !fout.good()){
    std::cerr << "file cannot be open: " << filename << std::endl;
  }
  for(std::size_t i = 0, I = dict.size(); i < I; ++i){
    fout << dict.get_key(i);
    for(std::size_t k = 0, K = embeddings.ncol(); k < K; ++k){
      fout << "\t" << embeddings[i][k];
    }
    fout << "\n";
  }
}


int main(int narg, char** argv)
{
  Arguments args = parse_args(narg, argv);

  std::string data_file = args.data_file;
  std::string result_embedding_file = args.result_embedding_file;

  Matrix<real> w_v;
  Matrix<real> w_u;
  std::vector<std::pair<std::size_t, std::size_t> > data;
  Dictionary<std::string> dict;
  Config<real> config;
  config.seed = args.seed;
  config.num_threads = args.num_threads;
  config.neg_size = args.neg_size;
  config.max_epoch = args.max_epoch;
  config.dim = args.dim;
  config.lr0 = args.lr0;
  config.lr1 = args.lr1;
  config.ws = args.ws;
  config.initializer = UniformInitializer<real>(-args.uniform_range, args.uniform_range);

  std::cout << "settings:" << "\n"
            << "  " << "data_file             : " << data_file << "\n"
            << "  " << "result_embedding_file : " << result_embedding_file << "\n"
            << "  " << "seed                  : " << config.seed << "\n"
            << "  " << "num_threads           : " << config.num_threads << "\n"
            << "  " << "neg_size              : " << config.neg_size << "\n"
            << "  " << "ws                    : " << config.ws << "\n"
            << "  " << "max_epoch             : " << config.max_epoch << "\n"
            << "  " << "dim                   : " << config.dim << "\n"
            << "  " << "lr0                   : " << config.lr0 << "\n"
            << "  " << "lr1                   : " << config.lr1 << "\n"
            << "  " << "uniform_range         : " << args.uniform_range << "\n"
            << std::endl;

  std::cout << "Creating token dictionary" << std::endl;
  
  bool ret = build_dict(dict, data_file, args);
  
  if(!ret){
    std::cerr << "file reading error" << std::endl;
    exit(1);
  }

  std::cout << "Done : " << dict.size() << " unique tokens." << std::endl;

  std::cout << "Starting training..." << std::endl;
  ret = poincare_embedding<real>(data_file, w_u, w_v, dict, config);

  if(!ret){
    std::cerr << "training failed" << std::endl;
    exit(1);
  }

  std::cout << "save to " << result_embedding_file << std::endl;
  save(result_embedding_file, w_v, dict);

  return 0;
}
