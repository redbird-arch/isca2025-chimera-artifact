#include "model/std_cells/NOR2.h"

#include <cmath>

#include "model/PortInfo.h"
#include "model/TransitionInfo.h"
#include "model/EventInfo.h"
#include "model/std_cells/StdCellLib.h"
#include "model/std_cells/CellMacros.h"
#include "model/timing_graph/ElectricalNet.h"
#include "model/timing_graph/ElectricalDriver.h"
#include "model/timing_graph/ElectricalLoad.h"
#include "model/timing_graph/ElectricalDelay.h"

namespace DSENT
{
    using std::ceil;
    using std::max;

    NOR2::NOR2(const String& instance_name_, const TechModel* tech_model_)
        : StdCell(instance_name_, tech_model_)
    {
        initProperties();
    }

    NOR2::~NOR2()
    {}

    void NOR2::initProperties()
    {
        return;
    }

    void NOR2::constructModel()
    {
        // All constructModel should do is create Area/NDDPower/Energy Results as
        // well as instantiate any sub-instances using only the hard parameters
        
        createInputPort("A");
        createInputPort("B");
        createOutputPort("Y");

        createLoad("A_Cap");
        createLoad("B_Cap");
        createDelay("A_to_Y_delay");
        createDelay("B_to_Y_delay");
        createDriver("Y_Ron", true);
        
        ElectricalLoad* a_cap = getLoad("A_Cap");
        ElectricalLoad* b_cap = getLoad("A_Cap");
        ElectricalDelay* a_to_y_delay = getDelay("A_to_Y_delay");
        ElectricalDelay* b_to_y_delay = getDelay("B_to_Y_delay");
        ElectricalDriver* y_ron = getDriver("Y_Ron");
        
        getNet("A")->addDownstreamNode(a_cap);
        getNet("B")->addDownstreamNode(b_cap);
        a_cap->addDownstreamNode(a_to_y_delay);        
        b_cap->addDownstreamNode(b_to_y_delay);        
        a_to_y_delay->addDownstreamNode(y_ron);
        b_to_y_delay->addDownstreamNode(y_ron);
        y_ron->addDownstreamNode(getNet("Y"));        
        
        // Create Area result
        // Create NDD Power result
        createElectricalAtomicResults();
        // Create NOR Event Energy Result
        createElectricalEventAtomicResult("NOR2");

        getEventInfo("Idle")->setStaticTransitionInfos();
        
        return;
    }
    
    void NOR2::updateModel()
    {
        // All updateModel should do is calculate numbers for the Area/NDDPower/Energy
        // Results as anything else that needs to be done using either soft or hard parameters

        // Get parameters
        double drive_strength = getDrivingStrength();
        Map<double>* cache = getTechModel()->getStdCellLib()->getStdCellCache();

        // Standard cell cache string
        String cell_name = "NOR2_X" + (String) drive_strength;

        // Get timing parameters
        getLoad("A_Cap")->setLoadCap(cache->get(cell_name + "->Cap->A"));
        getLoad("B_Cap")->setLoadCap(cache->get(cell_name + "->Cap->B"));        
        getDelay("A_to_Y_delay")->setDelay(cache->get(cell_name + "->Delay->A_to_Y"));
        getDelay("B_to_Y_delay")->setDelay(cache->get(cell_name + "->Delay->B_to_Y"));        
        getDriver("Y_Ron")->setOutputRes(cache->get(cell_name + "->DriveRes->Y"));
                
        // Set the cell area
        getAreaResult("Active")->setValue(cache->get(cell_name + "->ActiveArea"));
        getAreaResult("Metal1Wire")->setValue(cache->get(cell_name + "->ActiveArea"));
        
        return;
    }

    void NOR2::useModel()
    {
        // Get parameters
        double drive_strength = getDrivingStrength();
        Map<double>* cache = getTechModel()->getStdCellLib()->getStdCellCache();

        // Standard cell cache string
        String cell_name = "NOR2_X" + (String) drive_strength;

        // Propagate the transition info and get the 0->1 transtion count
        propagateTransitionInfo();
        double P_A = getInputPort("A")->getTransitionInfo().getProbability1();
        double P_B = getInputPort("B")->getTransitionInfo().getProbability1();
        double Y_num_trans_01 = getOutputPort("Y")->getTransitionInfo().getNumberTransitions01();

        // Calculate leakage
        double leakage = 0;
        leakage += cache->get(cell_name + "->Leakage->!A!B") * (1 - P_A) * (1 - P_B);
        leakage += cache->get(cell_name + "->Leakage->!AB") * (1 - P_A) * P_B;
        leakage += cache->get(cell_name + "->Leakage->A!B") * P_A * (1 - P_B);
        leakage += cache->get(cell_name + "->Leakage->AB") * P_A * P_B;
        getNddPowerResult("Leakage")->setValue(leakage);

        // Get VDD
        double vdd = getTechModel()->get("Vdd");
        // Get capacitances
        double y_cap = cache->get(cell_name + "->Cap->Y");
        double y_load_cap = getNet("Y")->getTotalDownstreamCap();                
        
        // Calculate NOR2Event energy
        double energy_per_trans_01 = (y_cap + y_load_cap) * vdd * vdd;
        getEventResult("NOR2")->setValue(energy_per_trans_01 * Y_num_trans_01);
                
        return;
    }
    
    void NOR2::propagateTransitionInfo()
    {
        // Get input signal transition info
        const TransitionInfo& trans_A = getInputPort("A")->getTransitionInfo();
        const TransitionInfo& trans_B = getInputPort("B")->getTransitionInfo();

        double max_freq_mult = max(trans_A.getFrequencyMultiplier(), trans_B.getFrequencyMultiplier());
        const TransitionInfo& scaled_trans_A = trans_A.scaleFrequencyMultiplier(max_freq_mult);
        const TransitionInfo& scaled_trans_B = trans_B.scaleFrequencyMultiplier(max_freq_mult);

        double A_prob_00 = scaled_trans_A.getNumberTransitions00() / max_freq_mult;
        double A_prob_01 = scaled_trans_A.getNumberTransitions01() / max_freq_mult;
        double A_prob_10 = A_prob_01;
        double A_prob_11 = scaled_trans_A.getNumberTransitions11() / max_freq_mult;
        double B_prob_00 = scaled_trans_B.getNumberTransitions00() / max_freq_mult;
        double B_prob_01 = scaled_trans_B.getNumberTransitions01() / max_freq_mult;
        double B_prob_10 = B_prob_01;
        double B_prob_11 = scaled_trans_B.getNumberTransitions11() / max_freq_mult;

        // Set output transition info
        double Y_prob_00 = A_prob_00 * B_prob_11 + 
                        A_prob_01 * (B_prob_10 + B_prob_11) +
                        A_prob_10 * (B_prob_01 + B_prob_11) +
                        A_prob_11;
        double Y_prob_01 = A_prob_00 * B_prob_10 + 
                        A_prob_10 * (B_prob_00 + B_prob_10);
        double Y_prob_11 = A_prob_00 * B_prob_00;
        
        // Check that probabilities add up to 1.0 with some finite tolerance
        ASSERT(LibUtil::DsentMath::isEqual((Y_prob_00 + Y_prob_01 + Y_prob_01 + Y_prob_11), 1.0), 
            "[Error] " + getInstanceName() +  "Output transition probabilities must add up to 1 (" +
            (String) Y_prob_00 + ", " + (String) Y_prob_01 + ", " + (String) Y_prob_11 + ")!");

        // Turn probability of transitions per cycle into number of transitions per time unit
        TransitionInfo trans_Y(Y_prob_00 * max_freq_mult, Y_prob_01 * max_freq_mult, Y_prob_11 * max_freq_mult);
        getOutputPort("Y")->setTransitionInfo(trans_Y);
        return;
    }
    
    void NOR2::cacheStdCell(StdCellLib* cell_lib_, double drive_strength_)
    {
        // Standard cell cache string
        String cell_name = "NOR2_X" + (String) drive_strength_;

        Log::printLine("=== " + cell_name + " ===");

        // Get parameters        
        double gate_pitch = cell_lib_->getTechModel()->get("Gate->PitchContacted");
        Map<double>* cache = cell_lib_->getStdCellCache();
        
        // Now actually build the full standard cell model
        // Create the two input ports
        createInputPort("A");
        createInputPort("B");
        createOutputPort("Y");
        
        // Adds macros
        CellMacros::addNor2(this, "NOR", true, true, true, "A", "B", "Y");
        CellMacros::updateNor2(this, "NOR", drive_strength_);

        // Cache area result
        double area = gate_pitch * getTotalHeight() * (1 + getGenProperties()->get("NOR_GatePitches").toDouble());
        cache->set(cell_name + "->ActiveArea", area);
        Log::printLine(cell_name + "->ActiveArea=" + (String) area);

        // --------------------------------------------------------------------
        // Leakage Model Calculation
        // --------------------------------------------------------------------
        double leakage_00 = getGenProperties()->get("NOR_LeakagePower_00").toDouble();
        double leakage_01 = getGenProperties()->get("NOR_LeakagePower_01").toDouble();        
        double leakage_10 = getGenProperties()->get("NOR_LeakagePower_10").toDouble();
        double leakage_11 = getGenProperties()->get("NOR_LeakagePower_11").toDouble();        
        cache->set(cell_name + "->Leakage->!A!B", leakage_00);
        cache->set(cell_name + "->Leakage->!AB", leakage_01);
        cache->set(cell_name + "->Leakage->A!B", leakage_10);
        cache->set(cell_name + "->Leakage->AB", leakage_11);
        Log::printLine(cell_name + "->Leakage->!A!B=" + (String) leakage_00);
        Log::printLine(cell_name + "->Leakage->!AB=" + (String) leakage_01);
        Log::printLine(cell_name + "->Leakage->A!B=" + (String) leakage_10);
        Log::printLine(cell_name + "->Leakage->AB=" + (String) leakage_11);
        // --------------------------------------------------------------------
        
        /*
        // Cache event energy results
        double event_a_flip = getGenProperties()->get("NOR_A1_Flip").toDouble();
        double event_b_flip = getGenProperties()->get("NOR_A2_Flip").toDouble();
        double event_zn_flip = getGenProperties()->get("NOR_ZN_Flip").toDouble();
        
        cache->set(cell_name + "->Event_A_Flip", event_a_flip);
        cache->set(cell_name + "->Event_B_Flip", event_b_flip);
        cache->set(cell_name + "->Event_ZN_Flip", event_zn_flip);
        Log::printLine(cell_name + "->Event_A_Flip=" + (String) event_a_flip);
        Log::printLine(cell_name + "->Event_B_Flip=" + (String) event_b_flip);
        Log::printLine(cell_name + "->Event_ZN_Flip=" + (String) event_zn_flip);
        */
        
        // --------------------------------------------------------------------
        // Get Node Capacitances
        // --------------------------------------------------------------------
        // Build abstracted timing model
        double a_cap = getNet("A")->getTotalDownstreamCap();
        double b_cap = getNet("B")->getTotalDownstreamCap();
        double y_cap = getNet("Y")->getTotalDownstreamCap();

        cache->set(cell_name + "->Cap->A", a_cap);
        cache->set(cell_name + "->Cap->B", b_cap);
        cache->set(cell_name + "->Cap->Y", y_cap);
        Log::printLine(cell_name + "->Cap->A_Cap=" + (String) a_cap);
        Log::printLine(cell_name + "->Cap->B_Cap=" + (String) b_cap);
        Log::printLine(cell_name + "->Cap->Y_Cap=" + (String) y_cap);
        // --------------------------------------------------------------------

        // --------------------------------------------------------------------
        // Build Internal Delay Model
        // --------------------------------------------------------------------
        double y_ron = getDriver("NOR_RonZN")->getOutputRes();
        double a_to_y_delay = getDriver("NOR_RonZN")->calculateDelay();
        double b_to_y_delay = getDriver("NOR_RonZN")->calculateDelay();
        
        cache->set(cell_name + "->DriveRes->Y", y_ron);
        cache->set(cell_name + "->Delay->A_to_Y", a_to_y_delay);
        cache->set(cell_name + "->Delay->B_to_Y", b_to_y_delay);
        Log::printLine(cell_name + "->DriveRes->Y=" + (String) y_ron);
        Log::printLine(cell_name + "->Delay->A_to_Y=" + (String) a_to_y_delay);
        Log::printLine(cell_name + "->Delay->B_to_Y=" + (String) b_to_y_delay);
        // --------------------------------------------------------------------
                
        return;

    }        

} // namespace DSENT

