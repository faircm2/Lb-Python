#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    int timestamp;
    double value;
} measurement_t;

void add_measurement(measurement_t **data, int *count, int *capacity)
{
    if (*count >= *capacity)
    {
        int new_capacity = (*capacity) * 2;

        measurement_t *new_data =
            realloc(*data, new_capacity * sizeof(measurement_t));

        if (new_data == NULL)
        {
            printf("Could not allocate memory.\n");
            return;
        }

        *data = new_data;
        *capacity = new_capacity;
    }

    printf("Enter timestamp: ");
    scanf("%d", &(*data)[*count].timestamp);

    printf("Enter value: ");
    scanf("%lf", &(*data)[*count].value);

    (*count)++;

    printf("Measurement added.\n");
}

void show_measurements(const measurement_t data[], int count)
{
    if (count == 0)
    {
        printf("There are no measurements.\n");
        return;
    }

    printf("\nMeasurements:\n");

    for (int i = 0; i < count; i++)
    {
        printf("%d: %.2f\n",
               data[i].timestamp,
               data[i].value);
    }
}

void sort_measurements(measurement_t data[], int count)
{
    for (int current = 1; current < count; current++)
    {
        measurement_t item = data[current];
        int position = current - 1;

        while (position >= 0 &&
               data[position].value > item.value)
        {
            data[position + 1] = data[position];
            position--;
        }

        data[position + 1] = item;
    }

    printf("Measurements sorted by value.\n");
}

void print_statistics(const measurement_t data[], int count)
{
    if (count == 0)
    {
        printf("There are no measurements.\n");
        return;
    }

    double smallest = data[0].value;
    double largest = data[0].value;
    double total = data[0].value;

    for (int i = 1; i < count; i++)
    {
        if (data[i].value < smallest)
        {
            smallest = data[i].value;
        }

        if (data[i].value > largest)
        {
            largest = data[i].value;
        }

        total += data[i].value;
    }

    printf("Minimum: %.2f\n", smallest);
    printf("Maximum: %.2f\n", largest);
    printf("Mean: %.2f\n", total / count);
}

void print_menu(void)
{
    printf("\n--- Measurement Program ---\n");
    printf("1 - Add measurement\n");
    printf("2 - Show measurements\n");
    printf("3 - Sort by value\n");
    printf("4 - Show statistics\n");
    printf("0 - Exit\n");
    printf("Selection: ");
}

int main(void)
{
    int capacity = 4;
    int count = 0;
    int choice;

    measurement_t *data =
        malloc(capacity * sizeof(measurement_t));

    if (data == NULL)
    {
        printf("Could not allocate memory.\n");
        return 1;
    }

    do
    {
        print_menu();
        scanf("%d", &choice);

        if (choice == 1)
        {
            add_measurement(&data, &count, &capacity);
        }
        else if (choice == 2)
        {
            show_measurements(data, count);
        }
        else if (choice == 3)
        {
            sort_measurements(data, count);
            show_measurements(data, count);
        }
        else if (choice == 4)
        {
            print_statistics(data, count);
        }
        else if (choice == 0)
        {
            printf("Program terminated.\n");
        }
        else
        {
            printf("Unknown option.\n");
        }

    } while (choice != 0);

    free(data);

    return 0;
}
