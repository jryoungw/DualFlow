#include "dsingle.h"
#include "math.h"

int
dsingle_isnonzero(dsingle q)
{
    return q.real != 0 && q.dual != 0;
}

int
dsingle_isnan(dsingle q)
{
    return isnan(q.real) || isnan(q.dual);
}

int
dsingle_isinf(dsingle q)
{
    return isinf(q.real) || isinf(q.dual);
}

int
dsingle_isfinite(dsingle q)
{
    return isfinite(q.real) && isfinite(q.dual);
}

float
dsingle_absolute(dsingle q)
{
   return sqrt(q.real*q.real + q.dual*q.dual);
}

dsingle
dsingle_add(dsingle q1, dsingle q2)
{
   return (dsingle) {
      q1.real+q2.real,
      q1.dual+q2.dual
   };
}

dsingle
dsingle_subtract(dsingle q1, dsingle q2)
{
   return (dsingle) {
      q1.real-q2.real,
      q1.dual-q2.dual
   };
}

dsingle
dsingle_multiply(dsingle q1, dsingle q2)
{
   return (dsingle) {
      q1.real*q2.real,
      q1.real*q2.dual + q1.dual*q2.real
   };
}

dsingle
dsingle_multiply_scalar(dsingle q, float s)
{
   return (dsingle) {s*q.real, s*q.dual};
}

dsingle
dsingle_divide_scalar(dsingle q, float s)
{
   return (dsingle) {q.real/s, q.dual/s};
}

dsingle
dsingle_divide(dsingle q1, dsingle q2)
{
   return (dsingle) {0 , 0}; // NOT IMPLEMENTED
}

dsingle
dsingle_log(dsingle q)
{
   return (dsingle) {0 , 0}; // NOT IMPLEMENTED
}

dsingle
dsingle_exp(dsingle q)
{
   return (dsingle) {0 , 0}; // NOT IMPLEMENTED
}

dsingle
dsingle_power(dsingle q, dsingle p)
{
   return (dsingle) {0 , 0}; // NOT IMPLEMENTED
}

dsingle
dsingle_power_scalar(dsingle q, float p)
{
   return (dsingle) {0 , 0}; // NOT IMPLEMENTED
}

dsingle
dsingle_negative(dsingle q)
{
   return (dsingle) {-q.real, -q.dual};
}

dsingle
dsingle_conjugate(dsingle q)
{
   return (dsingle) {q.real, -q.dual};
}

dsingle
dsingle_copysign(dsingle q1, dsingle q2)
{
    return (dsingle) {
        copysign(q1.real, q2.real),
        copysign(q1.dual, q2.dual)
    };
}

int
dsingle_equal(dsingle q1, dsingle q2)
{
    return 
        !dsingle_isnan(q1) &&
        !dsingle_isnan(q2) &&
        q1.real == q2.real && 
        q1.dual == q2.dual;
}

int
dsingle_not_equal(dsingle q1, dsingle q2)
{
    return !dsingle_equal(q1, q2);
}

int
dsingle_less(dsingle q1, dsingle q2)
{
    return
        (!dsingle_isnan(q1) &&
        !dsingle_isnan(q2)) && (
            q1.real != q2.real ? q1.real < q2.real :
            q1.dual != q2.dual ? q1.dual < q2.dual : 0);
}

int
dsingle_less_equal(dsingle q1, dsingle q2)
{
   return
        (!dsingle_isnan(q1) &&
        !dsingle_isnan(q2)) && (
            q1.real != q2.real ? q1.real < q2.real :
            q1.dual != q2.dual ? q1.dual < q2.dual : 1);
}
