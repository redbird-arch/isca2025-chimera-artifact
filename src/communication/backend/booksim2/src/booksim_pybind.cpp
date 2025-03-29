#ifdef LIB_BOOKSIM

#include <pybind11/pybind11.h>

#include "booksim_wrapper.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pybooksim, m) {
  m.doc() = "pybind11 booksim plugin";

  py::class_<BookSim>(m, "BookSim")
    .def(py::init<const std::string&>())
    .def("IssueMessage", &BookSim::IssueMessage)
    .def("PeekMessage", &BookSim::PeekMessage)
    .def("DequeueMessage", &BookSim::DequeueMessage)
    .def("SetSimTime", &BookSim::SetSimTime)
    .def("GetSimTime", &BookSim::GetSimTime)
    .def("GetMessageBufferSize", &BookSim::GetMessageBufferSize)
    .def("GetInjectBufferSize", &BookSim::GetInjectBufferSize)
    .def("Idle", &BookSim::Idle)
    .def("WakeUp", &BookSim::WakeUp)
    .def("CalculatePower", &BookSim::CalculatePower)
    .def("GetNetDynPower", &BookSim::GetNetDynPower)
    .def("GetNetLeakPower", &BookSim::GetNetLeakPower)
    .def("GetNetDynPowerWoClk", &BookSim::GetNetDynPowerWoClk)
    .def("GetRouterDynPower", &BookSim::GetRouterDynPower)
    .def("GetRouterLeakPower", &BookSim::GetRouterLeakPower)
    .def("GetBufferDynPower", &BookSim::GetBufferDynPower)
    .def("GetBufferLeakPower", &BookSim::GetBufferLeakPower)
    .def("GetXbarDynPower", &BookSim::GetXbarDynPower)
    .def("GetXbarLeakPower", &BookSim::GetXbarLeakPower)
    .def("GetSWAllocDynPower", &BookSim::GetSWAllocDynPower)
    .def("GetSWAllocLeakPower", &BookSim::GetSWAllocLeakPower)
    .def("GetClkDynPower", &BookSim::GetClkDynPower)
    .def("GetClkLeakPower", &BookSim::GetClkLeakPower)
    .def("GetLinkDynPower", &BookSim::GetLinkDynPower)
    .def("GetLinkLeakPower", &BookSim::GetLinkLeakPower)
    .def("GetNetLinkActivities", &BookSim::GetNetLinkActivities)
    ;

  py::class_<Message, std::unique_ptr<Message, py::nodelete>> message(m, "Message");

  py::enum_<Message::MessageType>(message, "MessageType")
    .value("ReadRequest", Message::MessageType::ReadRequest)
    .value("ReadReply", Message::MessageType::ReadReply)
    .value("WriteRequest", Message::MessageType::WriteRequest)
    .value("WriteReply", Message::MessageType::WriteReply)
    .value("ControlRequest", Message::MessageType::ControlRequest)
    .value("ControlReply", Message::MessageType::ControlReply)
    .value("ReduceData", Message::MessageType::ReduceData)
    .value("GatherData", Message::MessageType::GatherData)
    .value("MessageType_NUM", Message::MessageType::MessageType_NUM)
    .export_values();

  py::enum_<Message::SubMessageType>(message, "SubMessageType")
    .value("Head", Message::SubMessageType::Head)
    .value("Body", Message::SubMessageType::Body)
    .value("Tail", Message::SubMessageType::Tail)
    .value("HeadTail", Message::SubMessageType::HeadTail)
    .export_values();
};

#endif // #ifdef LIB_BOOKSIM
