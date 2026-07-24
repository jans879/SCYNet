#include <SCYNet.h>

#include <iostream>

using namespace std;

int main( int argc, char * argv[] )
{
  /** initialise SCYNet for 8TeV, load network */
  SCYNet scynet(8, argc, argv);
 
  /** define the parameter point */
  float x[11] = {-1682.23027931 , 3465.09031358 , 1237.59831623 , 2429.59820408 , 3450.31039446 ,2169.17885836 , 1800.49227919 , 1271.54318728, -4832.47899659  ,-610.85511464, 33.96309872};
  scynet.get_chi2(x);
  //change two parameter points
  x[2]=3000;
  x[3]=3000;
  //calculate chi2 again
  scynet.get_chi2(x);
}
