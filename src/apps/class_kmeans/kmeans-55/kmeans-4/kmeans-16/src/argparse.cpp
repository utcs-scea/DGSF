//
//  argparse.cpp
//  PS_Lab2_Test
//
//  Heejong Jang O'Keefe.
//  Adapted argparse.cpp file from Lab 1
//

// #include <argparse.h>
#include "argparse.h"

using namespace std;

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
   if (argc == 1)
   {
      std::cout << "Usage:" << std::endl;
      std::cout << "\t--n_cluster or -k <num_cluster>" << std::endl;
      std::cout << "\t--n_dims or -d <num_dims>" << std::endl;
      std::cout << "\t--in or -i <file_path>" << std::endl;
      std::cout << "\t--max_iter or -m <max_num_iter>" << std::endl;
      std::cout << "\t--threshold or -t <threshold>" << std::endl;
      std::cout << "\t[Optional] --centroid or -c" << std::endl;
      std::cout << "\t--seed or -s <seed>" << std::endl;
      exit(0);
   }

   opts->centroid = false;

   struct option l_opts[] = {
      {"n_cluster", required_argument, NULL, 'k'},
      {"n_dims", required_argument, NULL, 'd'},
      {"in", required_argument, NULL, 'i'},
      {"max_iter", required_argument, NULL, 'm'},
      {"threshold", required_argument, NULL, 't'},
      {"centroid", no_argument, NULL, 'c'},
      {"seed", required_argument, NULL, 's'}
   };

   int ind, c;
   //// Taking algorithm -a to choose which parallel algorithm to use? 
//    while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:a:c", l_opts, &ind)) != -1)
   while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:c", l_opts, &ind)) != -1)
   {
      switch (c)
      {
         case 0:
            break;
         case 'k':
            opts->n_cluster = atoi((char *)optarg);
            break;
         case 'd':
            opts->n_dims = atoi((char *)optarg);
            break;
         case 'i':
            opts->in_file = (char *)optarg;
            break;
         case 'm':
            opts->max_iter = atoi((char *)optarg);
            break;
         case 't':
            opts->threshold = stof((char *)optarg);
            break;
         case 'c':
            opts->centroid = true;
            break;
         case 's':
            opts->seed = atoi((char *)optarg);
            break;
         case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
      }
   }
}

