#ifndef __DSENT_MATH_H__
#define __DSENT_MATH_H__

#include <cmath>

namespace LibUtil
{
    class DsentMath
    {
        public:
            static const double epsilon;

            static inline bool isEqual(double value1_, double value2_)
            {
                return (std::fabs(value1_ - value2_) < epsilon);
            }
    };
} // namespace LibUtil

#endif // __MATH_H__

