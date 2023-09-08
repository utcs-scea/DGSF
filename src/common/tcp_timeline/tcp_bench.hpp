#ifndef __TCP_BENCH_HPP__
#define __TCP_BENCH_HPP__

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <sys/types.h>
#include <netinet/in.h>
#include <chrono>

#define THRESHOLD 2048
#define PORT 40057

enum Event {
    START,
    KERNEL,
    END,
};

extern std::vector<std::string> EventStrings;

struct EventLog {
    std::string logfile;
    std::string buffer;
    std::ofstream out;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;

    EventLog(std::string filename) : logfile(filename), out(filename) {
        buffer.reserve(THRESHOLD);
    }

    void append(std::string &text);
    void logAll();
    void log(const std::string &logtext);
    
};

struct TCPServer {
    int main_socket;
    fd_set fds;
    struct sockaddr_in address;
    int clients_accepted = 0;
    int clients_closed = 0;
    int n_clients;
    int* client_socket;
    EventLog log;

    TCPServer(int n_clients, std::string outfile);

    ~TCPServer();

    void init();
    void run();

private:
    void acceptClient();
    void handleEvent();    
    void recordEvent(struct sockaddr_in& addr, int event);
};

#endif
