#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 5555
#define BUFFER_SIZE 1024

int main(void)
{
    int sock;
    struct sockaddr_in server_address;
    char buffer[BUFFER_SIZE];

    sock = socket(AF_INET, SOCK_STREAM, 0);

    if (sock < 0)
    {
        perror("socket");
        return 1;
    }

    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(PORT);
    server_address.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sock,
                (struct sockaddr *)&server_address,
                sizeof(server_address)) < 0)
    {
        perror("connect");
        close(sock);
        return 1;
    }

    printf("Connected to server.\n");

    while (fgets(buffer, sizeof(buffer), stdin) != NULL)
    {
        send(sock, buffer, strlen(buffer), 0);

        ssize_t bytes = recv(sock, buffer, sizeof(buffer) - 1, 0);

        if (bytes <= 0)
        {
            printf("Server disconnected.\n");
            break;
        }

        buffer[bytes] = '\0';

        printf("Server: %s", buffer);
    }

    close(sock);

    return 0;
}