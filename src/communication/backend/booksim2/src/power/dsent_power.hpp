#ifndef _DSENT_POWER_HPP_
#define _DSENT_POWER_HPP_

#include "network.hpp"
#include "config_utils.hpp"
#include "flitchannel.hpp"
#include "DSENT.h"

class DSENTPower {

protected:
  // network under simulation
  Network * _net;
  int _classes;
  int _frequency;  // in MHz
  double _total_time;
  string _router_cfg;
  string _link_cfg;
  // write result to a tabbed format to file
  string _output_file_name;

  // DSENT instances
  DSENT::DSENT *_router_instance;
  DSENT::DSENT *_link_instance;

  // router dynamic power
  double _router_dyn_power;
  double _buf_wr_dyn_power;
  double _buf_rd_dyn_power;
  double _sw_arb_local_dyn_power;
  double _sw_arb_global_dyn_power;
  double _xbar_dyn_power;
  double _clk_dyn_power;

  // router static power
  double _router_leak_power;
  double _buf_leak_power;
  double _sw_arb_leak_power;
  double _xbar_leak_power;
  double _clk_leak_power;

  // link power
  double _link_dyn_power;
  double _link_leak_power;
  uint64_t _network_channel_activities;

  // network power
  double _net_dyn_power;
  double _net_leak_power;
  double _net_power;

  // methods
  void resetPowerStats();
  void calcRouterPower();
  void calcLinkPower();

public:
  DSENTPower(Network * net, const Configuration &config);
  ~DSENTPower();

  void Run();
  void CalculatePower();
  inline double GetNetworkPower() const { return _net_power; }
  inline double GetNetDynPower() const { return _net_dyn_power; }
  inline double GetNetLeakPower() const { return _net_leak_power; }
  inline double GetNetDynPowerWoClk() const { return _net_dyn_power - _clk_dyn_power; }
  inline double GetRouterDynPower() const { return _router_dyn_power; }
  inline double GetRouterLeakPower() const { return _router_leak_power; }
  inline double GetBufferDynPower() const { return _buf_wr_dyn_power + _buf_rd_dyn_power; }
  inline double GetBufferLeakPower() const { return _buf_leak_power; }
  inline double GetXbarDynPower() const { return _xbar_dyn_power; }
  inline double GetXbarLeakPower() const { return _xbar_leak_power; }
  inline double GetSWAllocDynPower() const { return _sw_arb_local_dyn_power + _sw_arb_global_dyn_power; }
  inline double GetSWAllocLeakPower() const { return _sw_arb_leak_power; }
  inline double GetClkDynPower() const { return _clk_dyn_power; }
  inline double GetClkLeakPower() const { return _clk_leak_power; }
  inline double GetLinkDynPower() const { return _link_dyn_power; }
  inline double GetLinkLeakPower() const { return _link_leak_power; }
  inline uint64_t GetNetLinkActivities() const { return _network_channel_activities; }

};

#endif
