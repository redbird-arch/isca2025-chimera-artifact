#include <fstream>
#include "booksim_wrapper.hpp"
#include "routefunc.hpp"
#include "random_utils.hpp"
#include "power_module.hpp"
#include <filesystem>

BookSim::BookSim(string const & configfile)
{
  _config = new BookSimConfig();
  _config->ParseFile(configfile);
  ifstream in(configfile);
  cout << endl << "Begin BookSim Configuration File: " << configfile << endl;
  while (!in.eof()) {
    char c;
    in.get(c);
    cout << c;
  }
  cout << "End BookSim Configuration File: " << configfile << endl;

  if (const char* srcpath = std::getenv("BOOKSIMSRC")) {
    _config->AddStrField("dsent_router_config", string(srcpath) +
        "/dsent/configs/dsent_router.cfg");
    _config->AddStrField("dsent_link_config", string(srcpath) +
        "/dsent/configs/dsent_link.cfg");
  } else {
    cout << "Error: Evironment not set up, source setup_env.sh!" << endl;
    exit(-1);
  }

  // initialize routing and global variables
  InitializeRoutingMap(*_config);

  gPrintActivity = (_config->GetInt("print_activity") > 0);
  gTrace = (_config->GetInt("viewer_trace") > 0);

  string watch_out_file = _config->GetStr("watch_out");
  if (watch_out_file == "") {
    gWatchOut = nullptr;
  } else if (watch_out_file == "-") {
    gWatchOut = &cout;
  } else {
    gWatchOut = new ofstream(watch_out_file.c_str());
  }

  // create network
  _net.push_back(nullptr);
  _net[0] = Network::New(*_config, "scale_network");

  // create traffic manager
  _traffic_manager = new ScaleTrafficManager(*_config, _net);

  _cur_mid = 0;
  _outstanding_messages = 0;
  _print_messages = gWatchOut && (_config->GetInt("watch_all_packets") > 0);

  _power_model = nullptr;
  if (_config->GetInt("sim_power") > 0 &&
      _config->GetInt("dsent_power") > 0) {
    _power_model = new DSENTPower(_net[0], *_config);
  }

#ifdef LIB_BOOKSIM
  trafficManager = _traffic_manager;
#endif
}

BookSim::~BookSim() {
  _traffic_manager->UpdateStats();
  _traffic_manager->DisplayStats();

  if (_config->GetInt("sim_power") > 0) {
    if (_config->GetInt("dsent_power") > 0) {
      assert(_power_model != nullptr);
      _power_model->Run();
    } else {
      Power_Module pnet(_net[0], *_config);
      pnet.run();
    }
  }

  delete _net[0];
  delete _config;
  delete _traffic_manager;
  if (_power_model) delete _power_model;
}

int BookSim::IssueMessage(int flow, int src, int dest, int id, int msg_size,
    Message::MessageType type, Message::SubMessageType subtype, int timestep,
    bool end)
{
  if (id == -1) {
    assert(subtype == Message::Head || subtype == Message::HeadTail);
    id = _cur_mid;
  }

  Message *message = Message::New(type, subtype, id, flow, src, dest, msg_size, timestep, end);
  if (_traffic_manager->Enqueue(message)) {
    _outstanding_messages++;
    // only increment the global message ID for head or head_tail submessage
    if (subtype == Message::Head || subtype == Message::HeadTail)
      _cur_mid++;
  } else {
    message->Free();
    return -1;
  }

  if (_print_messages) {
    *gWatchOut << GetSimTime() << " | node" << src << " | "
      << "Issuing message:" << endl;
   *gWatchOut << *message;
  }

  return id;
}

tuple<int, int, Message::MessageType, bool> BookSim::PeekMessage(int node, int vnet)
{
  int flow = -1;
  int src = -1;
  Message::MessageType type = Message::MessageType_NUM;
  bool end = false;

  Message *message = _traffic_manager->PeekMessage(node, vnet);
  if (message != nullptr) {
    flow = message->flow;
    src = message->src;
    type = message->type;
    end = message->end;

    if (_print_messages) {
      *gWatchOut << GetSimTime() << " | node" << node << " | "
        << "Receiving messages:" << endl;
     *gWatchOut << *message;
    }
  }

  return make_tuple(flow, src, type, end);
}

void BookSim::DequeueMessage(int node, int vnet)
{
  _traffic_manager->Dequeue(node, vnet);
  _outstanding_messages--;
}

bool BookSim::RunTest()
{
  _print_messages = true;

  vector<int> message_dests;
  vector<int> message_ids;
  vector<int> message_sizes;
  vector<Message::MessageType> message_types;
  message_ids.resize(gNodes, -1);
  message_dests.resize(gNodes, -1);
  message_sizes.resize(gNodes, 1024);
  message_types.resize(gNodes, Message::MessageType_NUM);

  for (int src = 0; src < gNodes; src++) {
    message_dests[src] = RandomInt(gNodes - 1);
    int mid = -1;
    Message::MessageType type = (Message::MessageType) RandomInt(Message::MessageType_NUM - 1);
    Message::SubMessageType subtype = Message::Head;
    mid = IssueMessage(8, src, message_dests[src], mid, 64, type, subtype, 0);
    assert(mid != -1);
    message_ids[src] = mid;
    message_types[src] = type;
    message_sizes[src] -= 64;
  }

  while (!_traffic_manager->Idle() || _outstanding_messages > 0) {
    _traffic_manager->WakeUp();
    for (int node = 0; node < gNodes; node++) {
      for (int vnet = 0; vnet < 2; vnet++) {
        tuple<int, int, Message::MessageType, bool> result = PeekMessage(node, vnet);
        if (get<1>(result) >= 0) DequeueMessage(node, vnet);
      }
    }

    for (int src = 0; src < gNodes; src++) {
      if (message_sizes[src] == 0)
        continue;

      int mid = message_ids[src];
      Message::MessageType type = message_types[src];
      Message::SubMessageType subtype;

      if (message_sizes[src] == 64) {
        subtype = Message::Tail;
      } else {
        subtype = Message::Body;
      }
      mid = IssueMessage(8, src, message_dests[src], mid, 64, type, subtype, 0);
      if (mid != -1) {
        message_sizes[src] -= 64;
      }
    }
  }

  _traffic_manager->UpdateStats();
  _traffic_manager->DisplayStats();

  return (_outstanding_messages == 0);
}

void BookSim::CalculatePower()
{
  if (_power_model)
    _power_model->CalculatePower();
}

double BookSim::GetNetDynPower()
{
  if (_power_model)
    return _power_model->GetNetDynPower();
  else
    return 0.0;
}

double BookSim::GetNetLeakPower()
{
  if (_power_model)
    return _power_model->GetNetLeakPower();
  else
    return 0.0;
}

double BookSim::GetNetDynPowerWoClk()
{
  if (_power_model)
    return _power_model->GetNetDynPowerWoClk();
  else
    return 0.0;
}

double BookSim::GetRouterDynPower()
{
  if (_power_model)
    return _power_model->GetRouterDynPower();
  else
    return 0.0;
}

double BookSim::GetRouterLeakPower()
{
  if (_power_model)
    return _power_model->GetRouterLeakPower();
  else
    return 0.0;
}

double BookSim::GetBufferDynPower()
{
  if (_power_model)
    return _power_model->GetBufferDynPower();
  else
    return 0.0;
}

double BookSim::GetBufferLeakPower()
{
  if (_power_model)
    return _power_model->GetBufferLeakPower();
  else
    return 0.0;
}

double BookSim::GetXbarDynPower()
{
  if (_power_model)
    return _power_model->GetXbarDynPower();
  else
    return 0.0;
}

double BookSim::GetXbarLeakPower()
{
  if (_power_model)
    return _power_model->GetXbarLeakPower();
  else
    return 0.0;
}

double BookSim::GetSWAllocDynPower()
{
  if (_power_model)
    return _power_model->GetSWAllocDynPower();
  else
    return 0.0;
}

double BookSim::GetSWAllocLeakPower()
{
  if (_power_model)
    return _power_model->GetSWAllocLeakPower();
  else
    return 0.0;
}

double BookSim::GetClkDynPower()
{
  if (_power_model)
    return _power_model->GetClkDynPower();
  else
    return 0.0;
}

double BookSim::GetClkLeakPower()
{
  if (_power_model)
    return _power_model->GetClkLeakPower();
  else
    return 0.0;
}

double BookSim::GetLinkDynPower()
{
  if (_power_model)
    return _power_model->GetLinkDynPower();
  else
    return 0.0;
}

double BookSim::GetLinkLeakPower()
{
  if (_power_model)
    return _power_model->GetLinkLeakPower();
  else
    return 0.0;
}

uint64_t BookSim::GetNetLinkActivities()
{
    if (_power_model)
      return _power_model->GetNetLinkActivities();
    else
      return 0;
}

