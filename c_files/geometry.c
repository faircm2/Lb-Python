#include "geometry.h"

#define PI 3.141592653589793

double circle_area(double r)
{
    return PI * r * r;
}

double rect_area(double w, double h)
{
    return w * h;
}