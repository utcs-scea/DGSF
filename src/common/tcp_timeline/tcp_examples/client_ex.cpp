#include <stdio.h>
#include <stdlib.h>
#include "tcp_bench.hpp"

int main(int argc, char* argv[]) {
    TCPClient client;
    client.notify(START);
    client.notify(END);
}
