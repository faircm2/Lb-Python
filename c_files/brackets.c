#include <stdio.h>
#include <stdlib.h>

#define INPUT_SIZE 1024

typedef struct StackNode
{
    char bracket;
    int index;
    struct StackNode *below;
} StackNode;

typedef struct
{
    StackNode *top;
} Stack;

/* Returns 1 if the stack is empty, otherwise 0 */
int is_empty(const Stack *stack)
{
    return stack->top == NULL;
}

/* Add a new character to the top of the stack */
int push(Stack *stack, char bracket, int index)
{
    StackNode *node = malloc(sizeof(*node));

    if (node == NULL)
    {
        return 0;
    }

    node->bracket = bracket;
    node->index = index;
    node->below = stack->top;

    stack->top = node;

    return 1;
}

/* Look at the top character without removing it */
char peek(const Stack *stack)
{
    if (is_empty(stack))
    {
        return '\0';
    }

    return stack->top->bracket;
}

/* Remove the top element */
char pop(Stack *stack)
{
    if (is_empty(stack))
    {
        return '\0';
    }

    StackNode *old_top = stack->top;
    char result = old_top->bracket;

    stack->top = old_top->below;

    free(old_top);

    return result;
}

/* Remove everything still stored in the stack */
void clear_stack(Stack *stack)
{
    while (!is_empty(stack))
    {
        pop(stack);
    }
}

/* Check if opening and closing brackets belong together */
int is_pair(char opening, char closing)
{
    return (opening == '(' && closing == ')') ||
           (opening == '[' && closing == ']') ||
           (opening == '{' && closing == '}');
}

/* Check the complete input line */
int check_brackets(const char text[])
{
    Stack stack = {NULL};
    int bracket_count = 0;

    for (int i = 0; text[i] != '\0'; i++)
    {
        char c = text[i];

        switch (c)
        {
        case '(':
        case '[':
        case '{':
            bracket_count++;

            if (!push(&stack, c, i))
            {
                printf("Memory allocation failed.\n");
                clear_stack(&stack);
                return 0;
            }

            break;

        case ')':
        case ']':
        case '}':
            bracket_count++;

            if (is_empty(&stack))
            {
                printf("Unexpected closing bracket '%c' at position %d.\n",
                       c, i);

                clear_stack(&stack);
                return 0;
            }

            if (!is_pair(peek(&stack), c))
            {
                printf("Mismatched bracket '%c' at position %d.\n",
                       c, i);

                clear_stack(&stack);
                return 0;
            }

            pop(&stack);
            break;
        }
    }

    if (!is_empty(&stack))
    {
        int position = stack.top->index;
        char bracket = stack.top->bracket;

        printf("Unclosed bracket '%c' at position %d.\n",
               bracket, position);

        clear_stack(&stack);
        return 0;
    }

    if (bracket_count == 0)
    {
        printf("No brackets found. The input is valid.\n");
    }
    else
    {
        printf("All brackets are balanced correctly.\n");
    }

    clear_stack(&stack);

    return 1;
}

int main(void)
{
    char text[INPUT_SIZE];

    printf("Enter a line: ");

    if (fgets(text, sizeof(text), stdin) == NULL)
    {
        printf("Empty input.\n");
        return 0;
    }

    check_brackets(text);

    return 0;
}
