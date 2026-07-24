
#include <string>

class SCYNet {
  /** 
   * \class SModelS
   * a simple class that wraps python-based SModelS for use with a C++ environment
   */

  public:
    SCYNet( int energy, int argc, char * argv[] );
    ~SCYNet();
    void get_chi2 ( const float x[] );

};

