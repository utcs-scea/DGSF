#ifdef DEBUG
#undef DEBUG
#define DEBUG(x) do { std::cerr << x << std::endl; } while (0)
#else
#define DEBUG(x)
#endif //DEBUG
