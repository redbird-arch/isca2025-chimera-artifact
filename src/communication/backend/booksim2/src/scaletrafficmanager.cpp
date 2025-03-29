#include <cmath>
#include <sstream>
#include <limits>
#include "scaletrafficmanager.hpp"

ScaleTrafficManager::ScaleTrafficManager(const Configuration &config, const
    vector<Network *> &net)
  : TrafficManager(config, net)
{
  _vnets = config.GetInt("vnets");
  _msg_buf_size = config.GetInt("msg_buf_size");
  _inject_buf_size = config.GetInt("inject_buf_size");
  _last_vnet.resize(_nodes, 0);
  _flit_size = config.GetInt("channel_width");
  _watch_all_packets = (config.GetInt("watch_all_packets") > 0);

  _sim_state = running;

  _input_buffer.resize(_nodes, vector<list<Message *> >());
  _output_buffer.resize(_nodes, vector<list<Message *> >());
  for (int node = 0; node < _nodes; node++) {
    _input_buffer[node].resize(_vnets, list<Message *>());
    _output_buffer[node].resize(_vnets, list<Message *>());
  }
}

ScaleTrafficManager::~ScaleTrafficManager()
{
  Message::FreeAll();
}

void ScaleTrafficManager::_RetireFlit(Flit *f, int dest)
{
  _deadlock_timer = 0;

  // send to the output message buffer
  if (f->tail || f->submsg_tail) {
    _output_buffer[dest][f->vnet].push_back(f->msg);
    if (f->watch) {
      *gWatchOut << GetSimTime() << " | " << FullName() << " | "
        << " HMC-" << dest
        << " consumes the subpacket " << f->subpid
        << " (packet " << f->pid
        << ") with submessage " << *(f->msg) << "." << endl;
    }
  }

  TrafficManager::_RetireFlit(f, dest);
}

void ScaleTrafficManager::_GeneratePacket(int source, int stype, int vnet, int time)
{
  int cl = 0;

  Message *message = _input_buffer[source][vnet].front();

  Flit::FlitType packet_type = Flit::ANY_TYPE;

  int size = (int) ceil((double) message->size*8 / _flit_size); // only count payload of submessage
  if (message->subtype == Message::Head || message->subtype == Message::HeadTail)
    size++;

  int packet_dest = message->dest;
  bool record = true;

  int timestep = message->timestep;

  bool watch = gWatchOut && (_packets_to_watch.count(_cur_pid) > 0);
  watch |= _watch_all_packets;

  switch (stype) {
    case Message::ReadRequest:
    case Message::WriteRequest:
    case Message::ControlRequest:
    case Message::ReduceData:
    case Message::GatherData:
      packet_type = Flit::REQUEST;
      break;
    case Message::ReadReply:
    case Message::WriteReply:
    case Message::ControlReply:
      packet_type = Flit::REPLY;
      break;
    default:
      ostringstream err;
      err << "Unknown message type " << stype;
      Error(err.str());
  }

  if (packet_dest < 0 || packet_dest >= _nodes) {
    ostringstream err;
    err << "Incorrect packet destination " << packet_dest
      << " for message type " << Message::MessageTypeToString(Message::MessageType(stype));
    Error(err.str());
  }

  int subnetwork = 0;

  if (watch) {
    *gWatchOut << GetSimTime() << " | "
      << "node" << source << " | "
      << "Enqueuing subpacket " << _cur_pid
      << " of packet " << message->id
      << " at time " << time
      << " through router " << source
      << " (to node " << packet_dest
      << " attached to router " << packet_dest
      << ") for submessage "
      << *message << endl;
  }

  int pid = message->id;
  int subpid = _cur_pid++;

  for (int i = 0; i < size; i++) {
    Flit * f = Flit::New();
    f->id = _cur_id++;
    assert(_cur_id);
    f->pid = pid;
    f->subpid = subpid;
    f->watch = watch | (gWatchOut && (_flits_to_watch.count(f->id) > 0));
    f->subnetwork = subnetwork;
    f->src = source;
    f->ctime = time;
    f->record = record;
    f->cl = cl;
    f->dest = packet_dest;
    f->type = packet_type;
    f->vnet = vnet;
    f->msg = message;

    _total_in_flight_flits[f->cl].insert(make_pair(f->id, f));
    if (record) {
      _measured_in_flight_flits[f->cl].insert(make_pair(f->id, f));
    }

    if (gTrace) {
      cout << "New Flit " << f->src << endl;
    }

    if (i == 0) { // Head flit
      if (message->subtype == Message::Head || message->subtype == Message::HeadTail) {
        f->head = true;
        f->dest = packet_dest;
      }
      f->size = size;
    } else {
      f->head = false;
      f->dest = -1;
    }
    switch (_pri_type) {
      case class_based:
        f->pri = cl;
        break;
      case age_based:
        f->pri = numeric_limits<int>::max() - time;
        break;
      case sequence_based:
        f->pri = numeric_limits<int>::max() - _packet_seq_no[source];
        break;
      case schedule_based:
        f->pri = numeric_limits<int>::max() - timestep;
        break;
      default:
        f->pri = 0;
    }
    assert(f->pri >= 0);

    if (i == (size - 1)) { // Tail flit
      if (message->subtype == Message::Tail || message->subtype == Message::HeadTail)
        f->tail = true;
      f->submsg_tail = true;
    } else {
      f->tail = false;
    }

    f->vc = -1;

    if (f->watch) {
      *gWatchOut << GetSimTime() << " | "
        << "node" << source << " | "
        << "Enqueuing flit " << f->id
        << " (subpacket " << f->subpid
        << ", packet " << f->pid
        << ") at time " << time
        << "." << endl;
    }

    _partial_packets[source][cl].push_back(f);
  }
  assert(_cur_pid);
}

void ScaleTrafficManager::_Inject()
{
  for (int input = 0; input < _nodes; input++) {
    assert(_classes == 1);

    int const last_vnet = _last_vnet[input];

    for (int v = 1; v <= _vnets; v++) {

      int vnet = (last_vnet + v) % _vnets;

      if (_input_buffer[input][vnet].empty() == false) {
        Message *message = _input_buffer[input][vnet].front();

        Message::MessageType packet_type = message->type;
        int size = (int) ceil((double) message->size*8 / _flit_size); // only count payload of submessage
        if (message->subtype == Message::Head || message->subtype == Message::HeadTail)
          size++;

        assert(_inject_buf_size == 0 || size <= _inject_buf_size);
        if (_inject_buf_size == 0 ||
            _partial_packets[input][0].size() + size <= (size_t) _inject_buf_size) {
          _GeneratePacket(input, packet_type, vnet, _time);
          if (_watch_all_packets) {
            *gWatchOut << GetSimTime() << " | " << FullName()
              << " | " << "HMC-" << input / gC << " (NI " << input % gC
              << ") to HMC-" << message->dest / gC << " (NI " << message->dest % gC << ")";
            if (message->subtype == Message::Head || message->subtype == Message::HeadTail) {
              *gWatchOut << " generate a new packet for message:" << endl;
            } else {
              *gWatchOut << " generate a new subpacket for message:" << endl;
            }
            *gWatchOut << *message;
          }
          _input_buffer[input][vnet].pop_front();
          _last_vnet[input] = vnet;
        }
      }
    }
  }
}

void ScaleTrafficManager::_Step()
{
  bool flits_in_flight = false;
  for (int c = 0; c < _classes; c++) {
    flits_in_flight |= !_total_in_flight_flits[c].empty();
  }
  if (flits_in_flight && (_deadlock_timer++ >= _deadlock_warn_timeout)) {
    _deadlock_timer = 0;
    cout << "WARNING: Possible network deadlock.\n";
    _DisplayRemaining(cout);
  }

  vector<map<int, Flit *> > flits(_subnets);

  for (int subnet = 0; subnet < _subnets; subnet++) {
    for (int n = 0; n < _nodes; n++) {
      Flit * const f = _net[subnet]->ReadFlit(n);
      if (f) {
        if (f->watch) {
          *gWatchOut << GetSimTime() << " | "
            << "node" << n << " | "
            << "Ejecting flit " << f->id
            << " (subpacket " << f->subpid
            << ", packet " << f->pid << ")"
            << " from VC " << f->vc
            << "." << endl;
        }
        flits[subnet].insert(make_pair(n, f));
        ++_accepted_flits[f->cl][n];
        if(f->tail) {
          ++_accepted_packets[f->cl][n];
        }
      }

      Credit * const c = _net[subnet]->ReadCredit(n);
      if (c) {
#ifdef TRACK_FLOWS
        for (set<int>::const_iterator iter = c->vc.begin(); iter != c->vc.end(); ++iter) {
          int const vc = *iter;
          assert(!_outstanding_classes[n][subnet][vc].empty());
          int cl = _outstanding_classes[n][subnet][vc].front();
          _outstanding_classes[n][subnet][vc].pop();
          assert(_outstanding_credits[cl][subnet][n] > 0);
          --_outstanding_credits[cl][subnet][n];
        }
#endif
        _buf_states[n][subnet]->ProcessCredit(c);
        c->Free();
      }
    }
    _net[subnet]->ReadInputs( );
  }

  _Inject();

  for (int subnet = 0; subnet < _subnets; subnet++) {
    for (int n = 0; n < _nodes; n++) {

      Flit * f = nullptr;
      BufferState * const dest_buf = _buf_states[n][subnet];

      int const last_class = _last_class[n][subnet];
      int class_limit = _classes;

      if (_hold_switch_for_packet) {
        list<Flit *> const & pp = _partial_packets[n][last_class];
        if (!pp.empty() && !pp.front()->head &&
            !dest_buf->IsFullFor(pp.front()->vc)) {
          f = pp.front();
          assert(f->vc == _last_vc[n][subnet][last_class]);

          // if we're holding the connection, we don't need to check that class
          // again in the for loop
          class_limit--;
        }
      }

      for (int i = 1; i <= class_limit; i++) {

        int const c = (last_class + i) % _classes;

        list<Flit *> const & pp = _partial_packets[n][c];

        if (pp.empty()) {
          continue;
        }

        Flit * const cf = pp.front();
        assert(cf);
        assert(cf->cl == c);

        if (cf->subnetwork != subnet) {
          continue;
        }

        if (f && (f->pri >= cf->pri)) {
          continue;
        }

        if (cf->head && cf->vc == -1) { // Find first available VC

          OutputSet route_set;
          _rf(nullptr, cf, -1, &route_set, true);
          set<OutputSet::sSetElement> const & os = route_set.GetSet();
          assert(os.size() == 1);
          OutputSet::sSetElement const & se = *os.begin();
          assert(se.output_port == -1);
          int vc_start = se.vc_start;
          int vc_end = se.vc_end;
          int vc_count = vc_end - vc_start + 1;
          if (_noq) {
            assert(_lookahead_routing);
            const FlitChannel * inject = _net[subnet]->GetInject(n);
            const Router * router = inject->GetSink();
            assert(router);
            int in_channel = inject->GetSinkPort();

            // NOTE: Because the lookahead is not for injection, but for the
            // first hop, we have to temporarily set cf's VC to be non-negative
            // in order to avoid seting of an assertion in the routing function.
            cf->vc = vc_start;
            _rf(router, cf, in_channel, &cf->la_route_set, false);
            cf->vc = -1;

            if (cf->watch) {
              *gWatchOut << GetSimTime() << " | "
                << "node" << n << " | "
                << "Generating lookahead routing info for flit " << cf->id
                << " (NOQ)." << endl;
            }
            set<OutputSet::sSetElement> const sl = cf->la_route_set.GetSet();
            assert(sl.size() == 1);
            int next_output = sl.begin()->output_port;
            vc_count /= router->NumOutputs();
            vc_start += next_output * vc_count;
            vc_end = vc_start + vc_count - 1;
            assert(vc_start >= se.vc_start && vc_start <= se.vc_end);
            assert(vc_end >= se.vc_start && vc_end <= se.vc_end);
            assert(vc_start <= vc_end);
          }
          if (cf->watch) {
            *gWatchOut << GetSimTime() << " | " << FullName() << " | "
              << "Finding output VC for flit " << cf->id
              << ":" << endl;
          }
          for (int i = 1; i <= vc_count; i++) {
            int const lvc = _last_vc[n][subnet][c];
            int const vc = (lvc < vc_start || lvc > vc_end) ?
              vc_start : (vc_start + (lvc - vc_start + i) % vc_count);
            assert(vc >= vc_start && vc <= vc_end);
            if (!dest_buf->IsAvailableFor(vc)) {
              if (cf->watch) {
                *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                  << "  Output VC " << vc << " is busy." << endl;
              }
            } else {
              if (dest_buf->IsFullFor(vc)) {
                if (cf->watch) {
                  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                    << "  Output VC " << vc << " is full." << endl;
                }
              } else if (_virtual_cut_through && dest_buf->AvailableFor(vc) < cf->size) {
                if (cf->watch) {
                  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                    << "  Output VC " << vc << " doesn't have enough buffer (cut-through)." << endl;
                }
              } else {
                if (cf->watch) {
                  *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                    << "  Selected output VC " << vc << "." << endl;
                }
                cf->vc = vc;
                break;
              }
            }
          }
        }

        if (cf->vc == -1) {
          if (cf->watch) {
            *gWatchOut << GetSimTime() << " | " << FullName() << " | "
              << "No output VC found for flit " << cf->id
              << "." << endl;
          }
        } else {
          if (dest_buf->IsFullFor(cf->vc)) {
            if (cf->watch) {
              *gWatchOut << GetSimTime() << " | " << FullName() << " | "
                << "Selected output VC " << cf->vc
                << " is full for flit " << cf->id
                << "." << endl;
            }
            // when buffer size is smaller than pacekt size, hold switch will
            // break and it will come here
            //assert("Jiayi: why it should be?" == nullptr);
          } else {
            f = cf;
          }
        }
      }

      if (f) {

        assert(f->subnetwork == subnet);

        int const c = f->cl;

        if (f->head) {

          if (_lookahead_routing) {
            if (!_noq) {
              const FlitChannel * inject = _net[subnet]->GetInject(n);
              const Router * router = inject->GetSink();
              assert(router);
              int in_channel = inject->GetSinkPort();
              _rf(router, f, in_channel, &f->la_route_set, false);
              if (f->watch) {
                *gWatchOut << GetSimTime() << " | "
                  << "node" << n << " | "
                  << "Generating lookahead routing info for flit " << f->id
                  << "." << endl;
              }
            } else if (f->watch) {
              *gWatchOut << GetSimTime() << " | "
                << "node" << n << " | "
                << "Already generated lookahead routing info for flit " << f->id
                << " (NOQ)." << endl;
            }
          } else {
            f->la_route_set.Clear();
          }

          dest_buf->TakeBuffer(f->vc);
          _last_vc[n][subnet][c] = f->vc;
        }

        _last_class[n][subnet] = c;

        _partial_packets[n][c].pop_front();

#ifdef TRACK_FLOWS
        ++_outstainding_credits[c][subnet][n];
        _outstanding_classes[n][subnet][f->vc].push(c);
#endif
        dest_buf->SendingFlit(f);

        if (_pri_type == network_age_based) {
          f->pri = numeric_limits<int>::max() - _time;
          assert(f->pri >= 0);
        }

        if (f->watch) {
          *gWatchOut << GetSimTime() << " | "
            << "node" << n << " | "
            << "Injecting flit " << f->id
            << " into subnet " << subnet
            << " at time " << _time
            << " with priority " << f->pri
            << "." << endl;
        }
        f->itime = _time;

        // Pass VC "back"
        if (!_partial_packets[n][c].empty() && !f->tail) {
          Flit * const nf = _partial_packets[n][c].front();
          nf->vc = f->vc;
        }

        ++_sent_flits[c][n];
        if (f->head) {
          ++_sent_packets[c][n];
        }

#ifdef TRACK_FLOWS
        ++_injected_flits[c][n];
#endif

        _net[subnet]->WriteFlit(f, n);
      }
    }
  }

  for (int subnet = 0; subnet < _subnets; subnet++) {
    for (int n = 0; n < _nodes; n++) {
      map<int, Flit *>::const_iterator iter = flits[subnet].find(n);
      if (iter != flits[subnet].end()) {
        Flit * const f = iter->second;

        f->atime = _time;
        if (f->watch) {
          *gWatchOut << GetSimTime() << " | "
            << "node" << n << " | "
            << "Injecting credit for VC " << f->vc
            << " into subnet " << subnet
            << "." << endl;
        }
        Credit * const c = Credit::New();
        c->vc.insert(f->vc);
        _net[subnet]->WriteCredit(c, n);

#ifdef TRACK_FLOWS
        ++_ejected_flits[f->cl][n];
#endif

        _RetireFlit(f, n);
      }
    }
    flits[subnet].clear();
    _net[subnet]->Evaluate();
    _net[subnet]->WriteOutputs();
  }

  ++_time;
  assert(_time);
  if (gTrace) {
    cout << "TIME" << _time << endl;
  }
}

bool ScaleTrafficManager::Enqueue(Message *message)
{
  int node = message->src;
  int vnet = message->vnet;

  if (_msg_buf_size == 0 ||
      _input_buffer[node][vnet].size() < (size_t) _msg_buf_size) {
    _input_buffer[node][vnet].push_back(message);
    return true;
  } else {
    return false;
  }
}

Message *ScaleTrafficManager::PeekMessage(int node, int vnet)
{
  Message *message = nullptr;

  if (!_output_buffer[node][vnet].empty()) {
    message = _output_buffer[node][vnet].front();
  }

  return message;
}

void ScaleTrafficManager::Dequeue(int node, int vnet)
{
  assert(!_output_buffer[node][vnet].empty());
  Message *message = _output_buffer[node][vnet].front();
  _output_buffer[node][vnet].pop_front();
  message->Free();
}
