#include <stdio.h>
#include "geometry.h"

int main(void)
{
    double radius = 8.0;
    double width = 6.0;
    double height = 4.0;

    printf("Circle area= %.2f\n", circle_area(radius));
    printf("Rectangle area= %.2f\n", rect_area(width, height));

    return 0;
}