#ifndef _SCALE_TRAFFICMANAGER_HPP_
#define _SCALE_TRAFFICMANAGER_HPP_

#include "message.hpp"
#include "trafficmanager.hpp"

class ScaleTrafficManager : public TrafficManager {

protected:
  int _flit_size;
  bool _watch_all_packets;
  int _vnets;
  int _msg_buf_size;
  int _inject_buf_size;
  vector<int> _last_vnet;

private:
  vector<vector<list<Message *> > > _input_buffer;
  vector<vector<list<Message *> > > _output_buffer;

protected:
  virtual void _RetireFlit(Flit *f, int dest);

  virtual void _Inject();
  virtual void _Step();

  virtual void _GeneratePacket(int source, int stype, int vnet, int time);

public:
  ScaleTrafficManager(const Configuration &config, const vector<Network *> &net);
  ~ScaleTrafficManager();

  bool Enqueue(Message *message);
  Message *PeekMessage(int node, int vnet);
  void Dequeue(int node, int vnet);

  inline void WakeUp() { _Step(); }
  inline bool Idle() {return (!Flit::OutStanding() && !Credit::OutStanding());}
};


#endif // _SCALETRAFFICMANAGER_HPP_
