#include "booksim.hpp"
#include <vector>
#include <sstream>
#include <cmath>

#include "bigraph.hpp"
#include "routefunc.hpp"


BiGraph::BiGraph(const Configuration &config, const string &name)
  : Network(config, name)
{
  _ComputeSize(config);
  _Alloc();
  _BuildNet(config);
}

void BiGraph::_ComputeSize(const Configuration &config)
{
  assert(gNumVCs % 2 == 0); // to break deadlock

  _m = config.GetInt("k");
  _n = config.GetInt("n");
  assert(_n % 2 == 0);

  gK = _m; gN = _n;

  _nodes = _m * _n;

  _size = _n;

  // (_n / 2) * _n
  _channels = (_n >> 1) * _n  ;
}

void BiGraph::_BuildNet(const Configuration &config)
{
#ifdef DEBUG_BIGRAPH
  cout << "BiGraph" << endl;
  cout << " M = " << _m << " sub-nodes per switch (physical group)" << endl;
  cout << " N = " << _n << " switches per virtual group" << endl;
  cout << " # of channels = " << _channels << endl;
  cout << " # of nodes " << _nodes << endl;
#endif

  // Number of routers at each part (Upper and Lower) of the network
  const int routers_per_part = _size / 2;

  // Allocate Routers
  string router_names[2] = {"upper_router", "lower_router"};
  ostringstream name;
  const int degree = _m + routers_per_part;
  int id = 0;
  for (int part = 0; part < 2; part++) {
    for (int pos = 0; pos < routers_per_part; pos++) {
      name.str("");
      name << router_names[part] << "_" << pos;
      _routers[id] = Router::NewRouter(config, this, name.str(), id, degree, degree);
      _timed_modules.push_back(_routers[id]);

      // connect injection and ejection channels
      for (int port = 0; port < _m; port++) {
        int link = id * _m + port;
        _routers[id]->AddInputChannel(_inject[link], _inject_cred[link]);
        _routers[id]->AddOutputChannel(_eject[link], _eject_cred[link]);
        _inject[link]->SetLatency(150);
        _inject_cred[link]->SetLatency(150);
        _eject[link]->SetLatency(150);
        _eject_cred[link]->SetLatency(150);
      }

      // connect output channels
      for (int port = 0; port < routers_per_part; port++) {
        int link = id * routers_per_part + port;
        _routers[id]->AddOutputChannel(_chan[link], _chan_cred[link]);
        _chan[link]->SetLatency(150);
        _chan_cred[link]->SetLatency(150);
        // TODO: parameterize latency
      }

      // connect input channels
      int part_link_base = (part == 0 ? _channels / 2 : 0);
      for (int port = 0; port < routers_per_part; port++) {
        int link = part_link_base + pos + port * routers_per_part;
        _routers[id]->AddInputChannel(_chan[link], _chan_cred[link]);
      }

      id++;
    }
  }
#ifdef DEBUG_BIGRAPH
  for (int i = 0; i < _size; i++) {
    cout << _routers[i]->Name() << " (id " << _routers[i]->GetID() << ")" << endl;
    int num_ports = _routers[i]->NumInputs();
    assert(num_ports == _routers[i]->NumOutputs());
    for (int port = 0; port < num_ports; port++) {
      const FlitChannel *channel = _routers[i]->GetOutputChannel(port);
      assert(channel);
      const Router *router = channel->GetSink();
      if (router) {
        cout << " to Router " << router->Name() << " (id " << router->GetID() << ") from port " << port << " through link " << channel->Name() << endl;
      } else {
        int link = i * _m + port;
        assert(_eject[link] == channel);
        cout << " to NI " << link << " from port " << port << " through link " << channel->Name() << endl;
      }
    }
  }
#endif
}

void BiGraph::RegisterRoutingFunctions() {
  gRoutingFunctionMap["dor_bigraph"] = &dor_bigraph;
}

void dor_bigraph(const Router *r, const Flit *f, int in_channel, OutputSet
    *outputs, bool inject) {
  int vcBegin = 0, vcEnd = gNumVCs - 1;
  if ( f->type == Flit::READ_REQUEST ) {
    vcBegin = gReadReqBeginVC;
    vcEnd = gReadReqEndVC;
  } else if ( f->type == Flit::WRITE_REQUEST ) {
    vcBegin = gWriteReqBeginVC;
    vcEnd = gWriteReqEndVC;
  } else if ( f->type ==  Flit::READ_REPLY ) {
    vcBegin = gReadReplyBeginVC;
    vcEnd = gReadReplyEndVC;
  } else if ( f->type ==  Flit::WRITE_REPLY ) {
    vcBegin = gWriteReplyBeginVC;
    vcEnd = gWriteReplyEndVC;
  } else if (f->type == Flit::REQUEST) {
    vcBegin = gRequestBeginVC;
    vcEnd = gRequestEndVC;
  } else if (f->type == Flit::REPLY) {
    vcBegin = gReplyBeginVC;
    vcEnd = gReplyEndVC;
  }
  assert(((f->vc >= vcBegin) && (f->vc <= vcEnd)) || (inject && (f->vc < 0)));

  int out_port;

  if (inject) {
    out_port = -1;
  } else {
    const int dest = f->dest;
    const int cur_router = r->GetID();
    const int dest_router = f->dest / gK;

    if (cur_router == dest_router) {
      out_port = dest % gK;
    } else {
      const int cur_router_part = cur_router / (gN / 2);
      const int dest_router_part = dest_router / (gN / 2);

      out_port = dest_router % (gN / 2) + gK;
      if (cur_router_part == dest_router_part && cur_router_part == 0) {
        vcBegin = gNumVCs / 2;
      } else if (cur_router_part == dest_router_part && cur_router_part == 1) {
        vcEnd = gNumVCs / 2 - 1;
      }
    }
  }

  if (f->watch) {
    if (!inject) {
      *gWatchOut << GetSimTime() << " | " << r->Name() << " | "
        << "Adding VC range [" << vcBegin << "," << vcEnd << "]"
        << " at output port " << out_port
        << " for flit " << f->id
        << " (input port " << in_channel
        << ", source " << f->src
        << ", source router " << f->src / gK
        << ", destination " << f->dest
        << ", destination router " << f->dest / gK << ")." << endl;
    } else {
      *gWatchOut << GetSimTime() << " | node" << f->src << " | (inject) "
        << "Adding VC range [" << vcBegin << "," << vcEnd << "]"
        << " at output port " << out_port
        << " for flit " << f->id
        << " (source " << f->src
        << ", source router " << f->src / gK
        << ", destination " << f->dest
        << ", destination router " << f->dest / gK << ")." << endl;
    }
  }

  outputs->Clear();
  outputs->AddRange(out_port, vcBegin, vcEnd);
}
