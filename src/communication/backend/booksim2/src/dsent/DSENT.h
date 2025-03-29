#ifndef __DSENT_DSENT_H__
#define __DSENT_DSENT_H__

// For DSENT operations
#include "libutil/OptionParser.h"
#include "libutil/Calculator.h"
#include "util/CommonType.h"
#include "util/Config.h"
#include "util/Result.h"
#include "model/Model.h"
#include "model/ModelGen.h"

// For timing optimization
#include "model/ElectricalModel.h"
#include "model/timing_graph/ElectricalNet.h"
#include "model/timing_graph/ElectricalTimingTree.h"
#include "model/timing_graph/ElectricalTimingOptimizer.h"
#include "model/PortInfo.h"

#include <string>
using std::string;

namespace DSENT
{
    using LibUtil::OptionParser;
    using LibUtil::Calculator;

    class DSENT
    {
        protected:
            class DSENTCalculator : public Calculator
        {
            public:
                DSENTCalculator(DSENT *dsent);
                virtual ~DSENTCalculator();
                double getExpr(const String& var_name_) const;

            protected:
                virtual double getEnvVar(const String& var_name_) const;

            private:
                DSENT *m_dsent_instance_;
        }; // class DSENTCalculator

        protected:
            void processQuery();
            const void* processQuery(const String& query_str_, bool is_print_);

            void performTimingOpt();

            void processEvaluate();

            // New content to integrate dsent into gem5
        public:
            DSENT(); // Jiayi
            DSENT(const String& model_name_);
            ~DSENT();
            static DSENT* getRouterInstance();
            static DSENT* getLinkInstance();
            static DSENT* createRouterInstance(const string router_cfg_);
            static DSENT* createLinkInstance(const string link_cfg_);
            void startNewCalculation();
            void evaluate();
            double queryResult(const String& var_name_);
            const StringMap* getEnvVarMap();
        protected:
            void init(const String& model_name_);
        private:
            DSENTCalculator *m_calc;
            static DSENT m_routerInstance;
            static DSENT m_linkInstance;
            bool ms_is_verbose_;
            Model* m_model;
            String m_evaluateString;
            StringMap *m_env_var;
    }; // class DSENT

} // namespace DSENT

#endif // __DSENT_DSENT_H__

