#include <stdio.h>
#include <stdlib.h>

/* Iterative binary search */
int search_iterative(int arr[], int length, int value)
{
    int low = 0;
    int high = length - 1;

    while (low <= high)
    {
        int mid = low + (high - low) / 2;

        if (arr[mid] == value)
        {
            return mid;
        }

        if (arr[mid] < value)
        {
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    }

    return -1;
}

/* Recursive binary search */
int search_recursive(int arr[], int low, int high, int value)
{
    if (low > high)
    {
        return -1;
    }

    int mid = low + (high - low) / 2;

    if (arr[mid] == value)
    {
        return mid;
    }

    if (value < arr[mid])
    {
        return search_recursive(arr, low, mid - 1, value);
    }

    return search_recursive(arr, mid + 1, high, value);
}

int main(int argc, char *argv[])
{
    int numbers[] = {4, 8, 15, 16, 23, 42, 50, 61};
    int length = sizeof(numbers) / sizeof(numbers[0]);

    if (argc != 2)
    {
        printf("Usage: %s <value>\n", argv[0]);
        return 1;
    }

    int value = atoi(argv[1]);

    int iterative_index = search_iterative(numbers, length, value);
    int recursive_index = search_recursive(numbers, 0, length - 1, value);

    printf("Searching for %d\n", value);

    if (iterative_index == -1)
    {
        printf("Iterative: number not found\n");
    }
    else
    {
        printf("Iterative: number found at index %d\n", iterative_index);
    }

    if (recursive_index == -1)
    {
        printf("Recursive: number not found\n");
    }
    else
    {
        printf("Recursive: number found at index %d\n", recursive_index);
    }

    return 0;
}
