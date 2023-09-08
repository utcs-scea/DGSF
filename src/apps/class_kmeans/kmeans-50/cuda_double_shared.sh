./bin/cuda_double -k 16 -d 32 -i input/random-n65536-d32-c16.txt -m 150 -t 0.00001 -s 8675309 -h 1 -c &> local_testing/d_cuda_shared.txt
./bin/cuda_double -k 16 -d 24 -i input/random-n16384-d24-c16.txt -m 150 -t 0.00001 -s 8675309 -h 1 -c &> local_testing/d_cuda_shared2.txt
./bin/cuda_double -k 16 -d 16 -i input/random-n2048-d16-c16.txt -m 150 -t 0.00001 -s 8675309 -h 1 -c &> local_testing/d_cuda_shared3.txt


./bin/cuda_double -k 16 -d 32 -i input/random-n65536-d32-c16.txt -m 150 -t 0.00001 -s 8675309 -h 1  &> local_testing/d_cuda_shared_c.txt
./bin/cuda_double -k 16 -d 24 -i input/random-n16384-d24-c16.txt -m 150 -t 0.00001 -s 8675309 -h 1  &> local_testing/d_cuda_shared_c2.txt
./bin/cuda_double -k 16 -d 16 -i input/random-n2048-d16-c16.txt -m 150 -t 0.00001 -s 8675309 -h 1  &> local_testing/d_cuda_shared_c3.txt
