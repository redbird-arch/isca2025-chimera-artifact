#include "dsent_power.hpp"
#include "booksim_config.hpp"
#include "switch_monitor.hpp"
#include "buffer_monitor.hpp"
#include "iq_router.hpp"

using LibUtil::String;

DSENTPower::DSENTPower(Network * n, const Configuration &config)
{

  _net = n;
  _output_file_name = config.GetStr("power_output_file");
  _classes = config.GetInt("classes");
  _frequency = config.GetInt("frequency"); // in MHz
  _router_cfg = config.GetStr("dsent_router_config");
  _link_cfg = config.GetStr("dsent_link_config");
  _total_time = 0;

  _router_instance = DSENT::DSENT::createRouterInstance(_router_cfg);
  _link_instance = DSENT::DSENT::createLinkInstance(_link_cfg);

  resetPowerStats();
}

DSENTPower::~DSENTPower() {
  delete _router_instance;
  delete _link_instance;
}

void DSENTPower::resetPowerStats()
{
  // router dynamic power
  _router_dyn_power = 0.0;
  _buf_wr_dyn_power = 0.0;
  _buf_rd_dyn_power = 0.0;
  _sw_arb_local_dyn_power = 0.0;
  _sw_arb_global_dyn_power = 0.0;
  _xbar_dyn_power = 0.0;
  _clk_dyn_power = 0.0;

  // router leakage power
  _router_leak_power = 0.0;
  _buf_leak_power = 0.0;
  _sw_arb_leak_power = 0.0;
  _xbar_leak_power = 0.0;
  _clk_leak_power = 0.0;

  // link power
  _link_dyn_power = 0.0;
  _link_leak_power = 0.0;
  _network_channel_activities = 0;

  // network power
  _net_dyn_power = 0.0;
  _net_leak_power = 0.0;
  _net_power = 0.0;
}

void DSENTPower::calcRouterPower()
{
  DSENT::DSENT * dsent = _router_instance;
  dsent->evaluate();

  vector<Router *> routers = _net->GetRouters();
  for (size_t i = 0; i < routers.size(); i++) {
    IQRouter * iq = dynamic_cast<IQRouter *>(routers[i]);
    const BufferMonitor * bm = iq->GetBufferMonitor();
    const SwitchMonitor * sm = iq->GetSwitchMonitor();

    // network activities
    const vector<int> reads = bm->GetReads();
    const vector<int> writes = bm->GetWrites();
    const vector<int> sw_activity = sm->GetActivity();
    double buf_reads = 0.0;
    double buf_writes = 0.0;
    double sw_acts = 0.0;

    for (int j = 0; j < bm->NumInputs()*_classes; j++) {
      buf_reads += reads[j];
      buf_writes += writes[j];
    }

    for (int j = 0; j < sm->NumInputs() * sm->NumOutputs() * _classes; j++) {
      sw_acts += sw_activity[j];
    }

    // buffer dynamic power
    _buf_wr_dyn_power += dsent->queryResult("buf_wr_dynamic") * (buf_writes/_total_time);
    _buf_rd_dyn_power += dsent->queryResult("buf_rd_dynamic") * (buf_reads/_total_time);

    // switch allcation local
    // Each input port chooses one input VC as requestor
    // Arbiter size: num_vclass*num_vc_per_vnet:1
    _sw_arb_local_dyn_power += dsent->queryResult("sa_o_dynamic_l") * (sw_acts/_total_time);

    // switch allocation global
    // Each output port chooses one input port as winner
    // Arbiter size: num_vnet*num_vc_per_vnet:1
    _sw_arb_global_dyn_power += dsent->queryResult("sa_o_dynamic_g") * (sw_acts/_total_time);

    // crossbar
    _xbar_dyn_power += dsent->queryResult("xbar_o_dynamic") * (sw_acts/_total_time);

    // clock dynamic power
    _clk_dyn_power += dsent->queryResult("clock_o_dynamic");

    // leakage power
    _buf_leak_power += dsent->queryResult("buffer_leakage");
    _sw_arb_leak_power += dsent->queryResult("sa_leakage");
    _xbar_leak_power += dsent->queryResult("xbar_leakage");
    _clk_leak_power += dsent->queryResult("clock_leakage");
  }

  // total power
  _router_dyn_power += _buf_wr_dyn_power + _buf_rd_dyn_power + _sw_arb_local_dyn_power + _sw_arb_global_dyn_power + _xbar_dyn_power + _clk_dyn_power;
  _router_leak_power += _buf_leak_power + _sw_arb_leak_power + _xbar_leak_power + _clk_leak_power;
}

void DSENTPower::calcLinkPower()
{
  DSENT::DSENT * dsent = _link_instance;
  dsent->evaluate();

  vector<FlitChannel *> inject = _net->GetInject();
  vector<FlitChannel *> eject = _net->GetEject();
  vector<FlitChannel *> chan = _net->GetChannels();

  // Assume 0.5 duty factor
  for (int i = 0; i < _net->NumNodes(); i++) {
    const vector<int> temp = inject[i]->GetActivity();
    double link_act = 0.0;
    for (int j = 0; j < _classes; j++)
      link_act += temp[j];
    _link_dyn_power += dsent->queryResult("link_dynamic") * (link_act / _total_time);
    _link_leak_power += dsent->queryResult("link_static");
  }

  for (int i = 0; i < _net->NumNodes(); i++) {
    const vector<int> temp = eject[i]->GetActivity();
    double link_act = 0.0;
    for (int j = 0; j < _classes; j++)
      link_act += temp[j];
    _link_dyn_power += dsent->queryResult("link_dynamic") * (link_act / _total_time);
    _link_leak_power += dsent->queryResult("link_static");
  }

  for (int i = 0; i < _net->NumChannels(); i++) {
    const vector<int> temp = chan[i]->GetActivity();
    //double link_act = 0.0;
    for (int j = 0; j < _classes; j++)
        _network_channel_activities += temp[j];
    //  link_act += temp[j];
    //_link_dyn_power += dsent->queryResult("link_dynamic") * (link_act / _total_time);
    //_link_leak_power += dsent->queryResult("link_static");
  }

}

void DSENTPower::CalculatePower()
{
  _total_time = GetSimTime();

  resetPowerStats();
  calcRouterPower();
  calcLinkPower();

  _net_dyn_power = _router_dyn_power + _link_dyn_power;
  _net_leak_power = _router_leak_power + _link_leak_power;
  _net_power = _net_dyn_power + _net_leak_power;
}

void DSENTPower::Run()
{
  CalculatePower();

  cout << "====== DSENT Power Summary ======" << endl;
  cout << "----- Complete time -----" << endl;
  cout << "   Cycles: " << _total_time << endl;
  cout << "----- Router -------" << endl;
  cout << "Buffer:" << endl;
  cout << "   Dynamic power: " << _buf_wr_dyn_power + _buf_rd_dyn_power << endl;
  cout << "   Leakage power: " << _buf_leak_power << endl;
  cout << "Crossbar:" << endl;
  cout << "   Dyanmic power: " << _xbar_dyn_power << endl;
  cout << "   Leakage power: " << _xbar_leak_power << endl;
  cout << "Switch allocator:" << endl;
  cout << "   Dyanmic power: " << _sw_arb_local_dyn_power + _sw_arb_global_dyn_power << endl;
  cout << "   Leakage power: " << _sw_arb_leak_power << endl;
  cout << "Clock:" << endl;
  cout << "   Dynamic power: " << _clk_dyn_power << endl;
  cout << "   Leakage power: " << _clk_leak_power << endl;
  cout << "Router Total: " << _router_dyn_power + _router_leak_power << endl;
  cout << "   Dynamic power: " << _router_dyn_power << endl;
  cout << "   Leakage power: " << _router_leak_power << endl;
  cout << "----- Link ------" << endl;
  cout << "Link Total: " << _link_dyn_power + _link_leak_power << endl;
  cout << "   Dynamic power: " << _link_dyn_power << endl;
  cout << "   Leakage power: " << _link_leak_power << endl;
  cout << "----- Network ------" << endl;
  cout << "Network Total: " << _net_power << endl;
  cout << "   Dynamic power: " << _net_dyn_power << endl;
  cout << "   Leakage power: " << _net_leak_power << endl;
  cout << "==============================" << endl;
}
