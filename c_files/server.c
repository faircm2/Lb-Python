#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 5555
#define BUFFER_SIZE 1024

int main(void)
{
    int server_fd;
    int client_fd;
    struct sockaddr_in address;
    char buffer[BUFFER_SIZE];

    server_fd = socket(AF_INET, SOCK_STREAM, 0);

    if (server_fd < 0)
    {
        perror("socket");
        return 1;
    }

    address.sin_family = AF_INET;
    address.sin_port = htons(PORT);
    address.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0)
    {
        perror("bind");
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 1) < 0)
    {
        perror("listen");
        close(server_fd);
        return 1;
    }

    printf("Server listening on 127.0.0.1:%d\n", PORT);

    client_fd = accept(server_fd, NULL, NULL);

    if (client_fd < 0)
    {
        perror("accept");
        close(server_fd);
        return 1;
    }

    while (1)
    {
        ssize_t bytes = recv(client_fd, buffer, sizeof(buffer) - 1, 0);

        if (bytes <= 0)
        {
            break;
        }

        buffer[bytes] = '\0';

        for (int i = 0; buffer[i] != '\0'; i++)
        {
            buffer[i] = toupper((unsigned char)buffer[i]);
        }

        send(client_fd, buffer, strlen(buffer), 0);
    }

    close(client_fd);
    close(server_fd);

    return 0;
}
