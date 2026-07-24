#include <SCYNet.h>
#include <Python.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

// https://docs.python.org/3/extending/embedding.html
// http://stackoverflow.com/questions/3286448/calling-a-python-method-from-c-c-and-extracting-its-return-value
//

SCYNet::SCYNet( int energy, int argc, char * argv[])
{
  cout << "initialize SCYNet at " <<energy<<" TeV"<<endl;

  /// initialise python, import modules
  wchar_t * program = Py_DecodeLocale( "SCYNet", NULL );
  if( program == NULL ){
    cerr << "Fatal error: cannot decode program name" << endl;
    exit(1);
  }
  Py_SetProgramName( program ); // intentionally not freed: Python keeps this pointer for the life of the interpreter

  Py_Initialize();

  // decode argv into wide strings for PySys_SetArgv
  std::vector<wchar_t*> wargv(argc);
  for( int i=0;i<argc;i++){
    wargv[i] = Py_DecodeLocale( argv[i], NULL );
  }
  PySys_SetArgv( argc, wargv.data() );
  for( auto w : wargv ) PyMem_RawFree(w); // PySys_SetArgv copies the strings into sys.argv, safe to free now

  // set the energy such that the python program can acess it.
  std::string var = "energy=" + std::to_string(energy);
  PyRun_SimpleString(var.c_str()); // now the variable energy is available in the python scripts

  //load network
  std::string load_net = "./load_network.py";
  FILE* fp = fopen( load_net.c_str(), "r" );
  if( fp ){
    PyRun_SimpleFileEx( fp, load_net.c_str(), 1 ); // 1 = fclose(fp) for us when done
  } else {
    cerr << "Could not open " << load_net << endl;
  }
  cout<<"Network loaded"<<endl;
  cout<<"--------------"<<endl;
}

SCYNet::~SCYNet ()
{
  Py_Finalize();
  cout << "SCYNet ends!" << endl;
}

void SCYNet::get_chi2 ( const float x[] )
{
  cout << "SCYNet for pMSSM-11 point " << x[0]<<" "<<x[1]<<" "<<x[2]<<" "<<x[3]<<" "<<x[4]<<" "<<x[5]<<" "<<x[6]<<" "<<x[7]<<" "<<x[8]<<" "<<x[9]<<" "<<x[10] << endl;

  //set x in the python environment (x in the python environment was already set in the load_network.py script
  for( int i=0;i<11;i++){
    std::string set_point = "x[0]["+std::to_string(i)+"]="+std::to_string(x[i]);
    PyRun_SimpleString( set_point.c_str());
  } 

  //calculate chi2
  std::string chi2_script = "./get_chi2.py";
  int t = -1;
  FILE* fp = fopen( chi2_script.c_str(), "r" );
  if( fp ){
    t = PyRun_SimpleFileEx( fp, chi2_script.c_str(), 1 ); // 1 = fclose(fp) for us when done
  } else {
    cerr << "Could not open " << chi2_script << endl;
  }
  cout<<"python script executes with "<<t<<endl;

  //read chi2 from file
  string line;
  ifstream myfile_1 ("chi2.txt");
  bool got_line = static_cast<bool>( getline (myfile_1,line) ); // istream's operator bool() is explicit since C++11

  cout <<"chi2 from SCYNET"<< line << endl;

  //one does not have to delete the "parameterpoint" which is in parameterpoint.txt, because the next call will anyway overwrite it
}
