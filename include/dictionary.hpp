 namespace poincare_disc { 

 ///////////////////////////////////////////////////////////////////////////////////////////
  // Utilities
  ///////////////////////////////////////////////////////////////////////////////////////////
  
    const std::string EOS = "</s>";
    const std::string BOW = "<";
    const std::string EOW = ">";
    
  template <class KeyType>
  struct Dictionary
  {
  public:
    using key_type = KeyType;
  public:
    Dictionary(): hash_(), keys_(), counts_() {}
    
    std::size_t size() const { return hash_.size(); }
    
    bool find(const key_type& key) const { return hash_.find(key) != hash_.end(); }
    
    std::size_t put(const key_type& key)
    {
        auto itr = hash_.find(key);
        if(itr == hash_.end()){
            std::size_t n = size();
            hash_.insert(std::make_pair(key, n));
            keys_.push_back(key);
            counts_.push_back(1);
            ++ntokens_;
            return n;
        } 
        std::size_t n = itr->second;
        ++counts_[n];
        ++ntokens_;
        return n;
    }
    
    std::size_t get_hash(const key_type& key) {
        if(!this->find(key)) {
            this->put(key);
        } 
        return hash_.find(key)->second;      
    }
    
    key_type get_key(const std::size_t i) const { return keys_[i]; }
    
    std::size_t get_count(const std::size_t i) const { return counts_[i]; }

    const std::vector<std::size_t>& counts() const { return counts_; }
    
    const std::size_t& tokenCount() const { return ntokens_; } 
    
    const bool readWord(std::istream& in, std::string& word) 
    {
        char c;
        std::streambuf& sb = *in.rdbuf();
        word.clear();
        
        while ((c = sb.sbumpc()) != std::char_traits<wchar_t>::eof()) {
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
                c == '\f' || c == '\0') {
                if (word.empty()) {
                    if (c == '\n') {
                        word += EOS;
                        return true;
                    }
                    continue;
                } else {
                    if (c == '\n')
                        sb.sungetc();
                    return true;
                }
            }
            word.push_back(c);
        }
        // trigger eofbit
        in.get();
        return !word.empty();
    }

    const int32_t getLine(std::istream& in, std::vector<int32_t>& words) {
        std::uniform_real_distribution<> uniform(0, 1);
        
        if (in.eof()) {
            in.clear();
            in.seekg(std::streampos(0));
        }

        words.clear();
        int32_t ntokens = 0;
        std::string token;
        while (this->readWord(in, token)) {
            size_t wid = get_hash(token);
            ntokens++;
            words.push_back(wid);       
            if (token == EOS) break;
        }
        return ntokens;
    }

    
  private:
    std::unordered_map<key_type, std::size_t> hash_;
    std::vector<key_type> keys_;
    std::vector<std::size_t> counts_;
    std::size_t ntokens_ = 0;
  };
  
    inline bool build_dict(Dictionary<std::string>& dict, const std::string& filename, const Arguments& args)
    {
        std::ifstream fin(filename.c_str());
        if(!fin || !fin.good()){
          std::cerr << "cannot read file: " << filename << std::endl;
          return false;
        }

        std::string word;
        size_t ntokens_ = 0;
        while (dict.readWord(fin, word)) {
            dict.put(word);
            if (ntokens_ % 1000000 == 0 && args.verbose == true) {
                std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
            }
            ++ntokens_;
        }

        fin.close();

        std::cerr << std::endl;

        return true;
    } 
  
 }