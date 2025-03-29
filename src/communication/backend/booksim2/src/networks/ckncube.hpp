#ifndef _CKNCUBE_HPP_
#define _CKNCUBE_HPP_

#include "network.hpp"
#include "routefunc.hpp"

class CKNCube : public Network {
private:
  bool _cmesh;

  int _k;
  int _n;
  int _c;

  void _ComputeSize(const Configuration &config);
  void _BuildNet(const Configuration &config);

  int _LeftChannel(int router, int dim);
  int _RightChannel(int router, int dim);

  int _LeftRouter(int router, int dim);
  int _RightRouter(int router, int dim);

public:
  CKNCube(const Configuration &config, const string &name, bool cmesh);

  int GetN() const;
  int GetK() const;
  int GetC() const;

  static void RegisterRoutingFunctions();

  static int NodeToRouter(int node);
  static int NodeToPort(int node);
};

//
// Routing Functions
//
void dor_next_ctorus(int cur_router, int dest_node, int in_port, int *out_port, int
    *partition, bool balance = false);

void dor_ctorus(const Router *r, const Flit *f, int in_channel, OutputSet
    *outputs, bool inject);

int dor_next_cmesh(int cur_router, int dest_node, bool descending = false);

void dim_order_cmesh(const Router *r, const Flit *f, int in_channel, OutputSet
    *outputs, bool inject);
#endif
