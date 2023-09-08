#ifndef __TCP_BENCH_CLIENT_HPP__
#define __TCP_BENCH_CLIENT_HPP__

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sys/types.h>
#include <netinet/in.h>
#include <string>
#include <arpa/inet.h>
#include <iostream>

#define PORT 50057

class TCPTimelineClient {
    int sock;
    public:
    void connect(std::string address = "") {
        struct sockaddr_in serv_addr;

        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            perror("Socket creation error");
        }

        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(PORT);

        // Convert IPv4 and IPv6 addresses from text to binary form
        std::string add = address.empty() ? "127.0.0.1" : address.c_str();
        if(inet_pton(AF_INET, add.c_str(), &serv_addr.sin_addr) <= 0) {
            std::cerr << "TCPTimelineClient: Invalid address/ Address not supported"
                << add << std::endl;
            return;
        }
    
        if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            std::cerr << "TCPTimelineClient: Connection to " << add << " failed. "
                << "Continuing without it." << std::endl;
            sock = -1;
        }
    }

    void notify(int e) {
        if (sock < 1) {
            //std::cerr << "TCPTimelineClient: Socket connection failed. Event not sent."
            //    << std::endl;
            return;
        }

        send(sock, &e, sizeof(int), 0);
    }
};


TCPTimelineClient ttc;

#endif
