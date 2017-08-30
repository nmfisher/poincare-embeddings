 #include <ios>

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Filestream Utils
  ///////////////////////////////////////////////////////////////////////////////////////////
  inline int64_t size(std::ifstream& ifs) {
    ifs.seekg(std::streamoff(0), std::ios::end);
    return ifs.tellg();
  }

  inline void seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::streampos(pos));
  }