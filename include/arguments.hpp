using real = double;

struct Arguments
{
  std::string data_file;
  std::string result_embedding_file = "embeddings.csv";
  unsigned int seed = 0;
  std::size_t num_threads = 1;
  std::size_t neg_size = 5;
  std::size_t ws = 5;
  std::size_t max_epoch = 1;
  std::size_t dim = 50;
  real uniform_range = 0.001;
  real lr0 = 0.01;
  real lr1 = 0.0001;
  bool verbose = true;
  ModelType model = ModelType::CBOW;
};

Arguments parse_args(int narg, char** argv)
{
  Arguments result;
  std::size_t arg_count = 0;
  std::string program_name = argv[0];
  for(int i = 1; i < narg; ++i){
    std::string arg(argv[i]);
    if(arg == "-s" || arg == "--seed"){
      arg = argv[++i];
      int n = std::stol(arg);
      if( n < 0 ){ goto HELP; }
      result.seed = static_cast<unsigned int>(n);
      continue;
    }else if(arg == "-t" || arg == "--num_thread"){
      arg = argv[++i];
      int n = std::stoi(arg);
      if( n <= 0 ){ goto HELP; }
      result.num_threads = static_cast<std::size_t>(n);
      continue;
    }else if(arg == "-n" || arg == "--neg_size"){
      arg = argv[++i];
      int n = std::stoi(arg);
      if( n <= 0 ){ goto HELP; }
      result.neg_size = static_cast<std::size_t>(n);
      continue;
    }else if(arg == "-e" || arg == "--max_epoch"){
      arg = argv[++i];
      int n = std::stoi(arg);
      if( n <= 0 ){ goto HELP; }
      result.max_epoch = static_cast<std::size_t>(n);
      continue;
    }else if(arg == "-d" || arg == "--dim"){
      arg = argv[++i];
      int n = std::stoi(arg);
      if( n <= 0 ){ goto HELP; }
      result.dim = static_cast<std::size_t>(n);
      continue;
    }else if(arg == "-v" || arg == "--verbose"){
      arg = argv[++i];
      int v = std::stoi(arg);
      if( v != 0 || v != 1 ){ goto HELP; }
      result.verbose = bool(v);
      continue;
    }else if(arg == "-l" || arg == "--learning_rate_init"){
      arg = argv[++i];
      double x = std::stod(arg);
      if( x <= 0 ){ goto HELP; }
      result.lr0 = static_cast<real>(x);
      continue;
    }else if(arg == "-L" || arg == "--learning_rate_final"){
      arg = argv[++i];
      double x = std::stod(arg);
      if( x <= 0 ){ goto HELP; }
      result.lr1 = static_cast<real>(x);
      continue;
    }else if(arg == "-u" || arg == "--uniform_range"){
      arg = argv[++i];
      double x = std::stod(arg);
      if( x <= 0 ){ goto HELP; }
      result.uniform_range = static_cast<real>(x);
      continue;
    }else if(arg == "-w" || arg == "--window_size"){
      arg = argv[++i];
      int x = std::stod(arg);
      if( x < 0 ){ goto HELP; }
      result.ws = static_cast<int>(x);
      continue;
    }else if(arg == "-m" || arg == "--model"){
      arg = argv[++i];
      if( !(arg == "cbow" || arg == "skipgram" )){ goto HELP; }
      if(arg == "cbow") {
        result.model = ModelType::CBOW;
      } else if(arg =="skipgram") {
        result.model = ModelType::SKIPGRAM;  
      } else {
          std::cerr << "Unrecognized model type." << std::endl;
      }
      continue;
    }else if(arg == "-h" || arg == "--help"){
      goto HELP;
    }

    if(arg_count == 0){
      result.data_file = arg;
      ++arg_count;
      continue;
    }else if(arg_count == 1){
      result.result_embedding_file = arg;
      ++arg_count;
      continue;
    }

    std::cerr << "invalid argument: " << arg << std::endl;
    goto HELP;
  }

  if(arg_count == 0){
    std::cerr << "missing argments" << std::endl;
    goto HELP;
  }else if(arg_count > 2){
    std::cerr << "too many argments" << std::endl;
    goto HELP;
  }

  return result;

 HELP:
  std::cerr <<
    "usage: " << program_name << " data_file [result_embedding_file] [options...]\n"
    "\n"
    "    data_file                 : string    input txt file \n"
    "    result_embeddng_file      : string    result file into which resulting embeddings are written\n"
    "    -s, --seed                : int >= 0   random seed\n"
    "    -t, --num_threads         : int > 0   number of threads\n"
    "    -m, --model               : string    model name (\"cbow\" (default) or \"skipgram\")\n"
    "    -n, --neg_size            : int > 0   negativ sample size\n"
    "    -e, --max_epoch           : int > 0   maximum training epochs\n"
    "    -d, --dim                 : int > 0   dimension of embeddings\n"
    "    -u, --uniform_range       : float > 0 embedding uniform initializer range\n"
    "    -l, --learning_rate_init  : float > 0 initial learning rate\n"
    "    -L, --learning_rate_final : float > 0 final learning rate\n"
    "    -v, --verbose             : int 0,1 verbosity \n"
    "    -w, --window_size        : int >= 0 window size for CBOW"
            << std::endl;
  exit(0);
}
