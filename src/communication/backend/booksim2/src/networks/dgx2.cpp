// $Id$

/*
 Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

////////////////////////////////////////////////////////////////////////
// DGX2 NVLINK Fabric 
// Description: 
//  Building Block:  
//    -->Each GPU connects to six siwtches 
//    -->No routing between switch planes required     
//    -->Eight NVLinks of the 18 available per switch are used to
//    connect to GPUs.     
//    -->Ten NVLinks available per switch to communicate outside the 
//    local group (level)(only eight is required for getting maximum bandwidth)
//  GPU clusters:
//    -->Two of the above building blocks together form a fully connected 
//    16 GPU cluster
//    -->non-blocking, non-iterfaring, unless same destination is involved
////////////////////////////////////////////////////////////////////////
// RCS Information:
//  Author: Pritam Majumder 
//  Texas A&M University  
//  $Date: 07/29/2020 3:06 am 
////////////////////////////////////////////////////////////////////////


#include "booksim.hpp"
#include <vector>
#include <sstream>
#include <cmath>

#include "dgx2.hpp"
#include "misc_utils.hpp"


#define DGX_DEBUG

DGX2::DGX2( const Configuration& config,const string & name )
  : Network( config ,name)
{
  _ComputeSize( config );
  _Alloc( );
  _BuildNet( config );
#ifdef DGX_DEBUG
  _PrintPortLinkMap();
#endif 
}

void DGX2::_ComputeSize( const Configuration& config )
{
  _k = config.GetInt( "k" );
  _n = config.GetInt( "n" );
  _c = config.GetInt( "c" );

  gK = _k; gN = _n; gC = _c;
  
  _nodes = _k * _n * _c;  
  _size = _k * _n;  
  _channels =  _c * _k * 2;  
  
  _input_port.resize(_k * _n);
  _output_port.resize(_k * _n);

  _inputChannelMap.resize(_k * _n);//number of routers _k * _n
  _outputChannelMap.resize(_k * _n);
  for(int router=0; router<_k*_n; router++){
    _inputChannelMap[router].resize(_c + _c);
    _outputChannelMap[router].resize(_c + _c);
  }
}


void DGX2::RegisterRoutingFunctions() {

}

void DGX2::_BuildNet( const Configuration& config )
{
  cout << "Fat Tree" << endl;
  cout << " k = " << _k << " levels = " << _n << endl;
  cout << " # of switches = "<<  _size << endl;
  cout << " # of channels = "<<  _channels << endl;
  cout << " # of nodes ( size of network ) = " << _nodes << endl;

  const int nPos =  _k;

  int link_latency;
  link_latency = config.GetInt("link_latency");

  //
  // Allocate Routers
  //
  ostringstream name;
  int level, pos, id, degree, port;
  for ( level = 0 ; level < _n ; level++) {
    for ( pos = 0 ; pos < nPos ; pos++) {

      degree = _c * 2;

      id = level * _k + pos;

      name.str("");
      name << "router_level" << level << "_" << pos;
      Router * r = Router::NewRouter( config, this, name.str( ), id,
          degree, degree );
      _Router( level, pos ) = r;
      _timed_modules.push_back(r);
    }
  }


  //
  // Connect Channels to Routers
  //

  // Connecting  Injection & Ejection Channels  
  
  for ( level = 0 ; level < _n ; level++ ) {
    for ( pos = 0 ; pos < nPos ; pos++ ) {
      for(int index = 0; index<_c; index++){
        //int link = pos*_k + index;
        int link = level*_c*_k +  pos +  index * _k; 
        
        //adding input channel and so as the inports
        _Router(level, pos)->AddInputChannel( _inject[link],
            _inject_cred[link]);
        int inport = _input_port[level * _k + pos];
        _inputChannelMap[level *_k + pos][inport] = link;
        _input_port[level * _k + pos]++;
        
        //adding output channel and so as the outports
        _Router(level, pos)->AddOutputChannel( _eject[link],
            _eject_cred[link]);
        int outport = _output_port[level * _k + pos];
        _outputChannelMap[level *_k + pos][outport] = link;
        _output_port[level * _k + pos]++;
        
        
        //setting the access latency 
        _inject[link]->SetLatency( link_latency );
        _inject_cred[link]->SetLatency( link_latency );
        _eject[link]->SetLatency( link_latency );
        _eject_cred[link]->SetLatency( link_latency );
      }
    }
  }

#ifdef DGX_DEBUG
  cout<<"\nAssigning  input and output\n";
#endif
  /*
   * OUTPUT iand INPUT channels
   */

  //down links from lower level to higher level
  //up links from from higher level to lower level
  //connect the input channel 
  //output channel for one router will input for another  
  
  for ( level = 0 ; level < _n ; level++) {
    for ( pos = 0 ; pos < nPos ; pos++) {
      for(int index = 0; index<_c; index++){

        int link = level * _c * _k + (pos * _c) + index;
        port = pos;
        int input_level = (level==0)? level + 1 : level - 1;
        
        assert(port<_k);
        assert(link<_channels);

        _Router(level, port)->AddOutputChannel( _chan[link],
            _chan_cred[link]);
        int outport = _output_port[level * _k + pos];
        _outputChannelMap[level *_k + pos][outport] = link;
        _output_port[level * _k + pos]++;

        _Router(input_level, port)->AddInputChannel( _chan[link],
            _chan_cred[link]);
        int inport = _input_port[level * _k + pos];
        _inputChannelMap[level *_k + pos][inport] = link;
        _input_port[level * _k + pos]++;

        _chan[link]->SetLatency( link_latency );
        _chan_cred[link]->SetLatency( link_latency ); 
      }
    }
  }  
}


Router*& DGX2::_Router( int depth, int pos ) 
{
  assert( depth < _n && pos < _k);
  return _routers[depth * _k + pos];
}

void DGX2::_PrintPortLinkMap(){
  //we are going to print router-wise
  cout<<"\n\n======================================================== "<<endl;
  cout<<"|||||||||   Printing Input/Output port mapping  ||||||| "<<endl;
  cout<<"========================================================= "<<endl;
  for(int router = 0; router < _k * _n; router++){
    cout<<"router-"<<router<<":"<<endl;
    for(int port = 0; port < (_c + _c); port++){
      if(port<_c){
        cout<<" inport:  "<<port<<" SW-2-GPU injection link: "<<_inputChannelMap[router][port]<<endl;
        cout<<" outport: "<<port<<" SW-2-GPU ejection link: "<<_outputChannelMap[router][port]<<endl;
      }
      else{
        cout<<" inport:  "<<port<<" SW-2-SW link: "<<_inputChannelMap[router][port]<<endl;
        cout<<" outport: "<<port<<" SW-2-SW link: "<<_outputChannelMap[router][port]<<endl;
      }
    }
    cout<<"--------------------------------------------------"<<endl;
  }
  cout<<"\n\n\n"<<endl;  
}
