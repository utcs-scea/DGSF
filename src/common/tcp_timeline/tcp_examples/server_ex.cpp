#include "tcp_bench.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {

    int n_clients = 1;
    std::string logfile;

    for (int i = 1; i < argc; i+=2) {
        if (strcmp(argv[i], "-f") == 0)
            logfile = argv[i+1];
        else if (strcmp(argv[i], "-n") == 0)
            n_clients = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-h") == 0) {
            printf("Usage: ./server_ex [-h] -n <num-clients> -f <log-filepath>\n");
            return 0;
        }
    }

    if (logfile.empty())
        logfile.assign("out.log");
    
    if (n_clients < 1) {
        printf("n_clients must be at least one\n");
        return -1;
    }

    TCPServer server(n_clients, logfile);

    server.init();
    server.run();

}
