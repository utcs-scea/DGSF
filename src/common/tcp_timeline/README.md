# Usage:

- Modify `tcp_bench.hpp/cpp` to contain desired Event enums and representative strings (for the eventlog). (There's probably a better way to get the string representation of the enum.)
- In the program you want to send events from, `#include tcp_bench.hpp`, declare a `TCPClient` object, then call `notify(Event e)` with the events you want. Call `notify(END)` when you want to close the client.
- Run a server by including `tcp_bench.hpp`, declaring a TCPServer object with number of clients that will be running, and a filepath for the eventlog (optional). call `init()`, then call `run()`. 
- Server will close when all n clients have called `notify(END)`.

# Compile tcp object files:

`make`

# Compile server and client examples:

`cd tcp_examples && make`

# Run examples:

`./tcp_examples/server_ex -n <num-clients> -f <logfile-path>`
`./tcp_examples/client_ex` OR `./tcp_examples/multiclient_ex <num-clients>`
