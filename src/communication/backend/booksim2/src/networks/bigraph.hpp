#ifndef _BiGraph_HPP_
#define _BiGraph_HPP_

#include "network.hpp"

class BiGraph : public Network {

  int _m; // NO. of logical groups, concentration of switch
  int _n; // NO. of physical groups = No. switches

  void _ComputeSize(const Configuration &config);
  void _BuildNet(const Configuration &config);

public:

  BiGraph(const Configuration &config, const string &name);
  static void RegisterRoutingFunctions();

};

void dor_bigraph(const Router *r, const Flit *f, int in_channel, OutputSet
    *outputs, bool inject = false);
#endif
