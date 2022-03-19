#ifndef __DSINGLE_H__
#define __DSINGLE_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	float real;
	float dual;
} dsingle;

int dsingle_isnonzero(dsingle q);
int dsingle_isnan(dsingle q);
int dsingle_isinf(dsingle q);
int dsingle_isfinite(dsingle q);
float dsingle_absolute(dsingle q);
dsingle dsingle_add(dsingle q1, dsingle q2);
dsingle dsingle_subtract(dsingle q1, dsingle q2);
dsingle dsingle_multiply(dsingle q1, dsingle q2);
dsingle dsingle_divide(dsingle q1, dsingle q2);
dsingle dsingle_multiply_scalar(dsingle q, float s);
dsingle dsingle_divide_scalar(dsingle q, float s);
dsingle dsingle_log(dsingle q);
dsingle dsingle_exp(dsingle q);
dsingle dsingle_power(dsingle q, dsingle p);
dsingle dsingle_power_scalar(dsingle q, float p);
dsingle dsingle_negative(dsingle q);
dsingle dsingle_conjugate(dsingle q);
dsingle dsingle_copysign(dsingle q1, dsingle q2);
int dsingle_equal(dsingle q1, dsingle q2);
int dsingle_not_equal(dsingle q1, dsingle q2);
int dsingle_less(dsingle q1, dsingle q2);
int dsingle_less_equal(dsingle q1, dsingle q2);

#ifdef __cplusplus
}
#endif

#endif
