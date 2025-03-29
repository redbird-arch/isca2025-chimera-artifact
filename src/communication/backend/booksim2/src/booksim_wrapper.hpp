#ifndef _BOOKSIM_WRAPPER_HPP_
#define _BOOKSIM_WRAPPER_HPP_

#include "booksim.hpp"
#include "message.hpp"
#include "booksim_config.hpp"
#include "network.hpp"
#include "scaletrafficmanager.hpp"
#include "dsent_power.hpp"

#include <string>

class BookSim {

public:
  BookSim(string const & configfile);
  ~BookSim();

  int IssueMessage(int flow, int src, int dest, int id, int msg_size,
      Message::MessageType type, Message::SubMessageType subtype, int timestep,
      bool end = true); // msg_size: bytes
  tuple<int, int, Message::MessageType, bool> PeekMessage(int node, int vnet);
  void DequeueMessage(int node, int vnet);
  void CalculatePower();
  double GetNetDynPower();
  double GetNetLeakPower();
  double GetNetDynPowerWoClk();
  double GetRouterDynPower();
  double GetRouterLeakPower();
  double GetBufferDynPower();
  double GetBufferLeakPower();
  double GetXbarDynPower();
  double GetXbarLeakPower();
  double GetSWAllocDynPower();
  double GetSWAllocLeakPower();
  double GetClkDynPower();
  double GetClkLeakPower();
  double GetLinkDynPower();
  double GetLinkLeakPower();
  uint64_t GetNetLinkActivities();

  bool RunTest();
  inline bool Idle() { return _traffic_manager->Idle() && _outstanding_messages == 0; }
  inline void WakeUp() { _traffic_manager->WakeUp(); }
  inline void SetSimTime(int time) { _traffic_manager->setTime(time); }
  inline int GetSimTime() { return _traffic_manager->getTime(); }
  inline int GetMessageBufferSize() { return _config->GetInt("msg_buf_size"); }
  inline int GetInjectBufferSize() { return _config->GetInt("inject_buf_size"); }

  inline void RegisterGlobalTrafficManager() {trafficManager = _traffic_manager;}

private:
  ScaleTrafficManager *_traffic_manager;
  BookSimConfig *_config;
  vector<Network *> _net;
  DSENTPower *_power_model;

  int _cur_mid;
  int _outstanding_messages;
  bool _print_messages;
};

#endif // _BOOKSIM_WRAPPER_HPP_

