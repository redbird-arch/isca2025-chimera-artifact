#include "DSENT.h"

#include <cstdlib>
#include <iostream>
#include <fstream>

namespace DSENT
{
    DSENT DSENT::m_routerInstance(string(getenv("BOOKSIMSRC")) + string("/dsent/configs/dsent_router.cfg"));
    DSENT DSENT::m_linkInstance(string(getenv("BOOKSIMSRC")) + string("/dsent/configs/dsent_link.cfg"));

    DSENT* DSENT::getRouterInstance() {
        return &m_routerInstance;
    }

    DSENT* DSENT::getLinkInstance() {
        return &m_linkInstance;
    }

    DSENT* DSENT::createRouterInstance(const string router_cfg_) {
        return new DSENT(String(router_cfg_));
    }

    DSENT* DSENT::createLinkInstance(const string link_cfg_) {
        return new DSENT(String(link_cfg_));
    }

    DSENT::DSENT() {
        m_model = NULL;
        m_env_var = NULL;
        m_calc = NULL;
        ms_is_verbose_ = false;
    }

    DSENT::DSENT(const String& model_name_) {
        m_model = NULL;
        m_env_var = NULL;
        m_calc = new DSENTCalculator(this);
        ms_is_verbose_ = false;
        init(model_name_);
    }

    DSENT::~DSENT() {
        delete m_model;
        delete m_calc;
        delete m_env_var;
    }

    void DSENT::init(const String& model_name_) {
        Config::allocate(model_name_);
        Config* dsent_config = Config::getSingleton();

        dsent_config->readString("");
        dsent_config->constructTechModel("");

        const String& model_name = dsent_config->get("ModelName");
        m_model = ModelGen::createModel(model_name, model_name, dsent_config->getTechModel());

        // Construct the model
        // Read all parameters the model requires
        const vector<String>* parameter_names = m_model->getParameterNames();
        // For all parameters, grab values from the config file
        for(vector<String>::const_iterator it = parameter_names->begin(); it != parameter_names->end(); ++it)
        {
            const String& parameter_name = *it;
            // If it exists in the config file, set the parameter
            if (dsent_config->keyExist(parameter_name))
            {
                m_model->setParameter(parameter_name, dsent_config->get(parameter_name));
            }
        }
        m_model->construct();

        // Update the model
        // Read all properties the model requires
        const vector<String>* property_names = m_model->getPropertyNames();
        // For all properties, grab values from the config file
        for(vector<String>::const_iterator it = property_names->begin(); it != property_names->end(); ++it)
        {
            const String& property_name = *it;
            // If it exists in the config file, set the parameter
            if(dsent_config->keyExist(property_name)) {
                m_model->setProperty(property_name, dsent_config->get(property_name));
            }
        }
        m_model->update();

        // Evaluate the model
        // Perform timing optimization if needed
        if(dsent_config->getIfKeyExist("IsPerformTimingOptimization", "false").toBool())
        {
            performTimingOpt();
        }
        m_evaluateString = dsent_config->get("EvaluateString");

        processQuery();

        // Copy all the key-values pairs of dsent config
        m_env_var = new StringMap(*dsent_config);
        Config::release();

        return;
    }

    void DSENT::evaluate() {
        processEvaluate();
    }

    void DSENT::startNewCalculation() {
        // Create a new calculator in order to recompute values of interest.
        delete m_calc;
        m_calc = new DSENTCalculator(this);
    }

    double DSENT::queryResult(const String& var_name_) {
        // Get expression result from the current calculator.
        return m_calc->getExpr(var_name_);
    }

    const StringMap* DSENT::getEnvVarMap() {
        return m_env_var;
    }

    void DSENT::processQuery()
    {
        Config* dsent_config = Config::getSingleton();
        vector<String> queries = dsent_config->get("QueryString").split(" ;\r\n");

        if(ms_is_verbose_)
        {
            cout << "Query results:" << endl;
            cout << "==============" << endl;
        }

        for(unsigned int i = 0; i < queries.size(); ++i)
        {
            const String& curr_query = queries[i];

            if(ms_is_verbose_)
            {
                String str = "Process query: '" + curr_query + "'";
                cout << str << endl;
                cout << String(str.size(), '-') << endl;
            }

            processQuery(curr_query, false);

            if(ms_is_verbose_)
            {
                cout << endl;
            }
        }
        if(ms_is_verbose_)
        {
            cout << "==============" << endl;
        }
        return;
    }

    const void* DSENT::processQuery(const String& query_str_, bool is_print_)
    {
        vector<String> type_split = query_str_.splitByString(Model::TYPE_SEPARATOR);
        ASSERT((type_split.size() == 2), "[Error] Invalid query format: " + query_str_);
        String query_type = type_split[0];

        vector<String> detail_split = type_split[1].splitByString(Model::DETAIL_SEPARATOR);
        ASSERT((detail_split.size() == 2), "[Error] Invalid query format: " + query_str_);
        String query_detail = detail_split[1];

        vector<String> subfield_split = detail_split[0].splitByString(Model::SUBFIELD_SEPARATOR);
        ASSERT(((subfield_split.size() == 2) || (subfield_split.size() == 1)), "[Error] Invalid query format: " + query_str_);
        String query_hier = subfield_split[0];
        String query_subfield = "";
        if(subfield_split.size() == 2)
        {
            query_subfield = subfield_split[1];
        }

        const void* query_result = m_model->parseQuery(query_type, query_hier, query_subfield);
        if(query_type == "Property")
        {
            const PropertyMap* property = (const PropertyMap*)query_result;
            if(is_print_)
            {
                cout << *property;
            }
        }
        else if(query_type == "Parameter")
        {
            const ParameterMap* parameter = (const ParameterMap*)query_result;
            if(is_print_)
            {
                cout << *parameter;
            }
        }
        else if(query_type.contain("Hier"))
        {
            const Model* model = (const Model*)query_result;
            if(is_print_)
            {
                model->printHierarchy(query_type, query_subfield, "", query_detail, cout);
            }
        }
        else
        {
            const Result* result = (const Result*)query_result;
            if(is_print_)
            {
                result->print(query_type + Model::TYPE_SEPARATOR + query_hier + 
                        Model::SUBFIELD_SEPARATOR + query_subfield, query_detail, cout);
            }
        }
        return query_result;
    }

    void DSENT::performTimingOpt()
    {
        Config* dsent_config = Config::getSingleton();

        // Get the frequency it is optimizing to
        double freq = dsent_config->get("Frequency").toDouble();

        // Get all the starting net names
        const vector<String>& start_net_names = dsent_config->get("TimingOptimization->StartNetNames").split("[,]");

        ASSERT((start_net_names.size() > 0), "[Error] Expecting net names in TimingOptimization->StartNetNames");

        if(start_net_names[0] == "*")
        {
            // Optimize from all input ports
            ElectricalModel* electrical_model = (ElectricalModel*)m_model;

            ElectricalTimingOptimizer timing_optimizer("Optimizer", electrical_model->getTechModel());
            timing_optimizer.setModel(electrical_model);
            timing_optimizer.construct();
            timing_optimizer.update();

            ElectricalTimingTree timing_tree(timing_optimizer.getInstanceName(), &timing_optimizer);

            const Map<PortInfo*>* input_ports = timing_optimizer.getInputs();
            Map<PortInfo*>::ConstIterator it_begin = input_ports->begin();
            Map<PortInfo*>::ConstIterator it_end = input_ports->end();
            Map<PortInfo*>::ConstIterator it;
            for(it = it_begin; it != it_end; ++it)
            {
                const String& net_name = it->first;
                timing_tree.performTimingOpt(timing_optimizer.getNet(net_name, makeNetIndex(0)), 1.0 / freq);
                //timing_tree.performTimingOpt(electrical_model->getNet(net_name, makeNetIndex(0)), 1.0 / freq);
            }
            // Loop the second times 
            for(it = it_begin; it != it_end; ++it)
            {
                //const String& net_name = it->first;
                //timing_tree.performTimingOpt(timing_optimizer.getNet(net_name, makeNetIndex(0)), 1.0 / freq);
            }
        }
        else
        {
            // TODO : parse the net name so that we could do hierarchical optimization
            // Currently we can only optimize timing at the top level
            ElectricalModel* electrical_model = (ElectricalModel*)m_model;
            ElectricalTimingTree timing_tree(electrical_model->getInstanceName(), electrical_model);
            for(unsigned int i = 0; i < start_net_names.size(); ++i)
            {
                const String& net_name = start_net_names[i];
                timing_tree.performTimingOpt(electrical_model->getNet(net_name), 1.0 / freq);
            }
        }
        return;
    }

    void DSENT::processEvaluate()
    {
        if(m_evaluateString == "") return;

        m_calc->evaluateString(m_evaluateString);

        return;
    }

    DSENT::DSENTCalculator::DSENTCalculator(DSENT *dsent)
    {
        m_dsent_instance_ = dsent;
    }

    DSENT::DSENTCalculator::~DSENTCalculator()
    {}

    double DSENT::DSENTCalculator::getExpr(const String& var_name_) const {
        return m_var_.getIfKeyExist(var_name_, 0.0);
    }

    double DSENT::DSENTCalculator::getEnvVar(const String& var_name_) const
    {
        if(m_var_.keyExist(var_name_))
        {
            return m_var_.get(var_name_);
        }
        else if(m_dsent_instance_->getEnvVarMap()->keyExist(var_name_))
        {
            return m_dsent_instance_->getEnvVarMap()->get(var_name_);
        }
        else
        {
            const Result* result = (const Result*)m_dsent_instance_->processQuery(var_name_ + "@0", false);
            return result->calculateSum();
        }
    }
} // namespace DSENT

