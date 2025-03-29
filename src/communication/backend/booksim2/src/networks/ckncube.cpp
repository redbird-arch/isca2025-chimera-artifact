#include "booksim.hpp"
#include <vector>
#include <sstream>
#include <ctime>
#include <cassert>
#include "ckncube.hpp"
#include "misc_utils.hpp"
#include "random_utils.hpp"


CKNCube::CKNCube(const Configuration &config, const string &name, bool cmesh)
  : Network(config, name)
{
  _cmesh = cmesh;

  _ComputeSize(config);
  _Alloc();
  _BuildNet(config);
}

void CKNCube::_ComputeSize(const Configuration &config)
{
  _k = config.GetInt("k");
  _n = config.GetInt("n");
  _c = config.GetInt("c");

  gK = _k;
  gN = _n;
  gC = _c;

  _size = powi(_k, _n);       // Number of routers in network
  _nodes = _c * _size;        // Number of nodes in network
  _channels = 2 * _n * _size; // Number of channels in network
}

void CKNCube::_BuildNet(const Configuration &config)
{
  int left_router;
  int right_router;

  int right_input;
  int left_input;

  int right_output;
  int left_output;

  ostringstream router_name;

  // latency type, noc or conventional network
  bool use_noc_latency = config.GetInt("use_noc_latency") == 1;

  for (int router = 0; router < _size; ++router) {
    router_name << "router";

    if (_k > 1) {
      for (int dim_offset = _size / _k; dim_offset >= 1; dim_offset /= _k) {
        router_name << "_" << (router / dim_offset) % _k;
      }
    }

    _routers[router] = Router::NewRouter(config, this, router_name.str(), router, 2*_n + _c, 2*_n + _c);
    _timed_modules.push_back(_routers[router]);

    router_name.str("");

    for (int dim = 0; dim < _n; ++dim) {
      // find the neighbor
      left_router = _LeftRouter(router, dim);
      right_router = _RightRouter(router, dim);

      // Current (R)outer
      // (L)eft router
      // (R)ight router
      //
      // L-->N<--R
      // L<--N-->R

      // torus channel is longer due to wrap around
      //int latency = _cmesh ? 1 : 2;
      int latency = 150; // TODO: parameterize

      // get the input channel number
      right_input = _LeftChannel(right_router, dim);
      left_input = _RightChannel(left_router, dim);

      // add the input channel
      _routers[router]->AddInputChannel(_chan[right_input], _chan_cred[right_input]);
      _routers[router]->AddInputChannel(_chan[left_input], _chan_cred[left_input]);

      // set input channel latency
      if (use_noc_latency) {
        _chan[right_input]->SetLatency(latency);
        _chan[left_input]->SetLatency(latency);
        _chan_cred[right_input]->SetLatency(latency);
        _chan_cred[left_input]->SetLatency(latency);
      } else {
        _chan[right_input]->SetLatency(1);
        _chan[left_input]->SetLatency(1);
        _chan_cred[right_input]->SetLatency(1);
        _chan_cred[left_input]->SetLatency(1);
      }

      // get the output channel
      right_output = _RightChannel(router, dim);
      left_output  = _LeftChannel(router, dim);

      //add the output channel
      _routers[router]->AddOutputChannel(_chan[right_output], _chan_cred[right_output]);
      _routers[router]->AddOutputChannel(_chan[left_output], _chan_cred[left_output]);

      //set output channel latency
      if(use_noc_latency){
        _chan[right_output]->SetLatency(latency);
        _chan[left_output]->SetLatency(latency);
        _chan_cred[right_output]->SetLatency(latency);
        _chan_cred[left_output]->SetLatency(latency);
      } else {
        _chan[right_output]->SetLatency(1);
        _chan[left_output]->SetLatency(1);
        _chan_cred[right_output]->SetLatency(1);
        _chan_cred[left_output]->SetLatency(1);
      }
    }

    // injection and ejection channel, always 1 latency
    for (int node = router*_c; node < (router+1)*_c; ++node) {
      _routers[router]->AddInputChannel(_inject[node], _inject_cred[node]);
      _routers[router]->AddOutputChannel(_eject[node], _eject_cred[node]);
      _inject[node]->SetLatency(1);
      _eject[node]->SetLatency(1);
      _inject_cred[node]->SetLatency(1);
      _eject_cred[node]->SetLatency(1);
    }
  }
}

int CKNCube::_LeftChannel(int router, int dim)
{
  // The base channel for a router is 2*_n*router
  int base = 2 * _n * router;
  // The offset for a left channel is 2*dim + 1
  int offset = 2 * dim + 1;

  return (base + offset);
}

int CKNCube::_RightChannel(int router, int dim)
{
  // The base channel for a router is 2*_n*router
  int base = 2 * _n * router;
  // The offset for a right channel is 2*dim
  int offset = 2 * dim;
  return (base + offset);
}

int CKNCube::_LeftRouter(int router, int dim)
{
  int k_to_dim = powi(_k, dim);
  int loc_in_dim = (router / k_to_dim) % _k;
  int left_router;
  // if at the left edge of the dimension, wraparound
  if (loc_in_dim == 0) {
    left_router = router + (_k-1) * k_to_dim;
  } else {
    left_router = router - k_to_dim;
  }

  return left_router;
}

int CKNCube::_RightRouter(int router, int dim)
{
  int k_to_dim = powi(_k, dim);
  int loc_in_dim = (router / k_to_dim) % _k;
  int right_router;
  // if at the right edge of the dimension, wraparound
  if (loc_in_dim == (_k-1)) {
    right_router = router - (_k-1) * k_to_dim;
  } else {
    right_router = router + k_to_dim;
  }

  return right_router;
}

int CKNCube::GetN() const
{
  return _n;
}

int CKNCube::GetK() const
{
  return _k;
}

int CKNCube::GetC() const
{
  return _c;
}


// ----------------------------------------------------------------------
//
//  Routing Helper Functions
//
// ----------------------------------------------------------------------

int CKNCube::NodeToRouter(int node)
{
  int router = node / gC;

  return router;
}

int CKNCube::NodeToPort(int node)
{
  // NI port base is 2*gN
  int base = 2 * gN;
  // NI port offset is node%gC
  int offset = node % gC;

  return (base + offset);
}

// ----------------------------------------------------------------------
//
//  Routing Functions
//
// ----------------------------------------------------------------------

void CKNCube::RegisterRoutingFunctions() {
  gRoutingFunctionMap["dor_ctorus"] = &dor_ctorus;
  gRoutingFunctionMap["dor_cmesh"] = &dim_order_cmesh;
}

// Concentrated Torus

void dor_next_ctorus(int cur_router, int dest_node, int in_port, int *out_port, int
    *partition, bool balance)
{
  int dim_left;
  int dir;
  int dist2;

  int dest_router = CKNCube::NodeToRouter(dest_node);

  for (dim_left = 0; dim_left < gN; ++dim_left) {
    if ((cur_router % gK) != (dest_router % gK)) {
      break;
    }
    cur_router /= gK;
    dest_router /= gK;
  }

  if (dim_left < gN) {
    if (in_port / 2 != dim_left) {
      // Turning into a new dimension

      cur_router %= gK;
      dest_router %= gK;
      dist2 = gK - 2 * ((dest_router - cur_router + gK) % gK);

      if ((dist2 > 0) ||
          ((dist2 == 0) && RandomInt(1))) {
        *out_port = 2 * dim_left; // Right
        dir = 0;
      } else {
        *out_port = 2 * dim_left + 1; // Left
        dir = 1;
      }

      if (partition) {
        if (balance) {
          // Cray's "Partition" allocation
          // Two datelines: one between k-1 and 0 which forces VC 1
          //                another between ((k-1)/2) and ((k-1)/2 + 1) which
          //                forces VC 0 otherwise any VC can be used

          if ((dir == 0 && cur_router > dest_router) ||
              (dir == 1 && cur_router < dest_router)) {
            *partition = 1;
          } else if (((dir == 0) && (cur_router <= (gK - 1)/2) && (dest_router > (gK - 1)/2)) ||
              ((dir == 1) && (cur_router > (gK - 1)/2) && (dest_router <= (gK - 1)/2))) {
            *partition = 0;
          } else {
            *partition = RandomInt(1); // use either VC set
          }
        } else {
          // Deterministic, fixed dateline between router k-1 and 0

          if ((dir == 0 && cur_router > dest_router) ||
              (dir == 1 && cur_router < dest_router)) {
            *partition = 1;
          } else {
            *partition = 0;
          }
        }
      }
    } else {
      // Inverting the least significant bit keeps
      // the packet moving in the same direction
      *out_port = in_port ^ 0x1;
    }
  } else {
    // ejection
    *out_port = CKNCube::NodeToPort(dest_node);
  }
}

void dor_ctorus(const Router *r, const Flit *f, int in_channel, OutputSet
    *outputs, bool inject)
{
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
    int cur_router = r->GetID();
    int dest = f->dest;
    int dest_router = CKNCube::NodeToRouter(dest);

    dor_next_ctorus(cur_router, dest, in_channel, &out_port, &f->ph, false);

    // at the destination router, we don't need to separate VCs by ring partition (dateline)
    if (cur_router != dest_router) {
      int const available_vcs = (vcEnd - vcBegin + 1) / 2;
      assert(available_vcs > 0);

      if (f->ph == 0) {
        vcEnd -= available_vcs;
      } else {
        vcBegin += available_vcs;
      }
    }
  }

  if (f->watch) {
    if (!inject) {
      *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
        << "Adding VC range [" << vcBegin << "," << vcEnd << "]"
        << " at output port " << out_port
        << " for flit " << f->id
        << " (input port " << in_channel
        << ", source " << f->src
        << ", source router " << CKNCube::NodeToRouter(f->src)
        << ", destination " << f->dest
        << ", destination router " << CKNCube::NodeToRouter(f->dest) << ")." << endl;
    } else {
      *gWatchOut << GetSimTime() << " | node" << f->src << " | (inject) "
        << "Adding VC range [" << vcBegin << "," << vcEnd << "]"
        << " at output port " << out_port
        << " for flit " << f->id
        << " (source " << f->src
        << ", source router " << CKNCube::NodeToRouter(f->src)
        << ", destination " << f->dest
        << ", destination router " << CKNCube::NodeToRouter(f->dest) << ")." << endl;
    }
  }

  outputs->Clear();
  outputs->AddRange(out_port, vcBegin, vcEnd);
}

int dor_next_cmesh(int cur_router, int dest_node, bool descending)
{
  int out_port = -1;

  int dest_router = CKNCube::NodeToRouter(dest_node);

  if (cur_router == dest_router) {
    out_port = CKNCube::NodeToPort(dest_node);
  } else {
    int cur = cur_router;
    int dest = dest_router;

    int dim_left;

    if(descending) {
      for ( dim_left = ( gN - 1 ); dim_left > 0; --dim_left ) {
        if ( ( cur * gK / gNodes ) != ( dest * gK / gNodes ) ) { break; }
        cur = (cur * gK) % gNodes; dest = (dest * gK) % gNodes;
      }
      cur = (cur * gK) / gNodes;
      dest = (dest * gK) / gNodes;
    } else {
      for ( dim_left = 0; dim_left < ( gN - 1 ); ++dim_left ) {
        if ( ( cur % gK ) != ( dest % gK ) ) { break; }
        cur /= gK; dest /= gK;
      }
      cur %= gK;
      dest %= gK;
    }

    if ( cur < dest ) {
      out_port = 2*dim_left;     // Right
    } else {
      out_port = 2*dim_left + 1; // Left
    }
  }

  return out_port;
}

void dim_order_cmesh(const Router *r, const Flit *f, int in_channel, OutputSet
    *outputs, bool inject)
{
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
    int cur_router = r->GetID();
    int dest = f->dest;

    out_port = dor_next_cmesh(cur_router, dest, false);
  }

  if (f->watch) {
    if (!inject) {
      *gWatchOut << GetSimTime() << " | " << r->FullName() << " | "
        << "Adding VC range [" << vcBegin << "," << vcEnd << "]"
        << " at output port " << out_port
        << " for flit " << f->id
        << " (input port " << in_channel
        << ", source " << f->src
        << ", source router " << CKNCube::NodeToRouter(f->src)
        << ", destination " << f->dest
        << ", destination router " << CKNCube::NodeToRouter(f->dest) << ")." << endl;
    } else {
      *gWatchOut << GetSimTime() << " | node" << f->src << " | (inject) "
        << "Adding VC range [" << vcBegin << "," << vcEnd << "]"
        << " at output port " << out_port
        << " for flit " << f->id
        << " (source " << f->src
        << ", source router " << CKNCube::NodeToRouter(f->src)
        << ", destination " << f->dest
        << ", destination router " << CKNCube::NodeToRouter(f->dest) << ")." << endl;
    }
  }

  outputs->Clear();
  outputs->AddRange(out_port, vcBegin, vcEnd);
}
