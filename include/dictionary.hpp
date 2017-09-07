  namespace poincare_disc { 

 ///////////////////////////////////////////////////////////////////////////////////////////
  // Utilities
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  const std::string EOS = "</s>";
  const std::string UNK = "<UNK>";
    
  template <class KeyType>
  struct Dictionary
  {
  public:
    using key_type = KeyType;
  public:
    Dictionary(): hash_(), keys_(), counts_() { }
    
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
    
    std::size_t* get_hash(const key_type& key) {
        if(this->find(key)) {
            return &hash_.find(key)->second;          
        } 
        return nullptr;
    }
    
    void threshold(size_t threshold, bool excludeFirst) {
        size_t num_removed = 0;
        key_type k;
        size_t i;
        if(excludeFirst) {
            i = 1;
        } else {
            i = 0;
        }

        while(i < counts_.size()) {         
            k = keys_[i];
            if(counts_[i] < threshold) {
                hash_.erase(k);
                counts_.erase(counts_.begin() + i);
                keys_.erase(keys_.begin() + i);
                num_removed++;
            } else {
                auto it = hash_.find(k);
                it->second = it->second - num_removed;
                ++i;
            }
        }
        keys_.shrink_to_fit();
        counts_.shrink_to_fit();
        
        for(i = 0; i < counts_.size(); i++) {
            k = keys_[i];
            std::cout << k << std::endl;
            std::cout << hash_.find(k)->second << std::endl;
            std::cout << counts_[i] << std::endl;
        }
    }
    
    key_type get_key(const std::size_t i) const { return keys_[i]; }
    
    std::size_t get_count(const std::size_t i) const { return counts_[i]; }

    const std::vector<std::size_t>& counts() const { return counts_; }
    
    const std::size_t& tokenCount() const { return ntokens_; } 
    
    const bool readWord(std::ifstream& ifs, std::string& word) 
    {
        char c;
        word.clear();
        std::streambuf& sb = *ifs.rdbuf();
        while ((c = sb.sbumpc()) != EOF) {
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
        return !word.empty();
    }

    const int32_t getLine(std::ifstream& ifs, std::vector<int32_t>& words) {
        std::uniform_real_distribution<> uniform(0, 1);

        words.clear();
        int32_t ntokens = 0;
        std::string token;
        while (this->readWord(ifs, token) && token != EOS) {
            size_t* wid = get_hash(token);
            ntokens++;
            if(wid) {
                words.push_back(*wid);       
            } else {
                wid = get_hash(UNK);
                words.push_back(*wid);       
            }
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
        std::ifstream ifs(filename.c_str());
        if(!ifs || !ifs.good()){
          std::cerr << "cannot read file: " << filename << std::endl;
          return false;
        }
        
        std::string word;
        size_t ntokens_ = 0;
        dict.put(UNK);
        while (dict.readWord(ifs, word)) {
            dict.put(word);
            if (ntokens_ % 1000000 == 0 && args.verbose == true) {
                std::cerr << "\rRead " << ntokens_  / 1000000 << "M words" << std::flush;
            }
            ++ntokens_;
        }

        ifs.close();

        std::cerr << std::endl;
        
        dict.threshold(args.threshold, true); // keep UNK

        return true;
    } 
  
 }