#include "tcp_bench.hpp"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <iostream>
#include <fstream>

std::vector<std::string> EventStrings = {
    "START",
    "SET_DEVICE",
    "CREATE_CTX",
    "PRECREATE_HANDLES",
    "SETSIGNALS_AND_LISTENPORT",
    "CREATE_TCPCHAN_LOG",
    "INIT_CMD_HANDLER",
    "WAIT_FOR_CLIENT",
    "CONN_RECEIVED",
    "CMD_HANDLER_CREATED",
    "CUBIN_LOADED_APP_START",
    "APP_FINISHED",
    "CLEANED_UP",
};

/////// EVENT LOG //////////

void EventLog::append(std::string &text) {
    using std::chrono::high_resolution_clock;
    using std::chrono::time_point_cast;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    auto now = high_resolution_clock::now();

    if (start.time_since_epoch().count() == 0) {
        start = now;
    }
    
    auto timestamp = duration_cast<milliseconds>(now - start).count();
    std::string line = std::to_string(timestamp) + ", " + text;

    if (buffer.length() + line.size() + 1 >= THRESHOLD) {
            logAll();
            buffer.resize(0);
    }
    buffer.append(line);
    buffer.append(1, '\n');
}

void EventLog::logAll() {
    out << buffer;
    buffer.resize(0);
}

void EventLog::log(const std::string &text) {
    out << text;
}

/////// TCP SERVER //////////

TCPServer::TCPServer(int n_clients, std::string outfile) : n_clients(n_clients), log(outfile) {
    client_socket = (int*)calloc(n_clients, sizeof(int));
}

TCPServer::~TCPServer() {
    free(client_socket);
}

void TCPServer::init() {

    int opt = 1;  
         
    if((main_socket = socket(AF_INET, SOCK_STREAM, 0)) == 0) {  
        perror("socket failed");  
        exit(-1);  
    }  
     
    // set main socket to allow multiple connections
    if(setsockopt(main_socket, SOL_SOCKET, SO_REUSEADDR, 
            &opt, sizeof(opt)) < 0 ) {  
        perror("setsockopt");  
        exit(-1);  
    }  
     
    address.sin_family = AF_INET;  
    address.sin_addr.s_addr = INADDR_ANY;  
    address.sin_port = htons(PORT);  
         
    if (bind(main_socket, (struct sockaddr*)&address, sizeof(address)) < 0) {  
        perror("!!!!!!!!!!!!!!!!! bind failed");  
        exit(-1);  
    }  

    printf("Listening on port %d \n", PORT);  
         
    if (listen(main_socket, n_clients) < 0) {  
        perror("listen");  
        exit(-1);        
    }  
           
}
 
void TCPServer::run() {
    int activity, sd, max_sd;  

    while (clients_closed != n_clients) {  
        FD_ZERO(&fds);  
        FD_SET(main_socket, &fds);  
        max_sd = main_socket;  
         
        //add child sockets to set 
        for (int i = 0; i < n_clients; i++) {  
            sd = client_socket[i];  
            if (sd > 0)  
                FD_SET(sd, &fds);  
            if (sd > max_sd)  
                max_sd = sd;  
        }  

        // wait for an activity on one of the sockets
        activity = select(max_sd+1, &fds, NULL, NULL, NULL);  
       
        if ((activity < 0) && (errno != EINTR)) {  
            printf("select error");  
        }  

        // incoming connection
        if (FD_ISSET(main_socket, &fds)) {  
            acceptClient();
        } else {
            handleEvent();
        }
    }

    printf("Shutting down server\n");
    log.logAll();
} 

void TCPServer::acceptClient() {
    int new_socket;
    int addrlen = sizeof(address);
    if ((new_socket = accept(main_socket, 
            (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {  
        perror("accept");  
        exit(-1);  
    }  
     
    printf("New connection, socket fd: %d, ip: %s, port: %d\n",
        new_socket, inet_ntoa(address.sin_addr), ntohs(address.sin_port));  

    client_socket[clients_accepted++] = new_socket; 
}

void TCPServer::recordEvent(struct sockaddr_in& addr, int event) {
    //std::string line = inet_ntoa(addr.sin_addr);
    //line += "::"; 
    //line += std::to_string(ntohs(addr.sin_port));
    //line += (" ");
    //line += EventStrings[event];
    std::string line = EventStrings[event];

    log.append(line);
}

void TCPServer::handleEvent() {

    int sd, nread;
    int addrlen = sizeof(address);
    int message;

    for (int i = 0; i < n_clients; i++) {  
        sd = client_socket[i];  
        if (FD_ISSET(sd, &fds)) {  
            nread = read(sd, &message, sizeof(message));
            // close connection
            if (nread == 0) {  
                getpeername(sd, (struct sockaddr*)&address,
                        (socklen_t*)&addrlen);  
                printf("Host disconnected, ip %s, port %d \n", 
                        inet_ntoa(address.sin_addr), ntohs(address.sin_port));  
                close(sd);  
                client_socket[i] = 0;
                clients_closed++;
            } else {  
                recordEvent(address, message);
            }  
        }  
    }        
}

