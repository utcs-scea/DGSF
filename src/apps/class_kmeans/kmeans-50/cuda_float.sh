./bin/cuda_float -k 16 -d 32 -i input/random-n65536-d32-c16.txt -m 150 -t 0.00001 -s 8675309 -c &> local_testing/f_cuda.txt
./bin/cuda_float -k 16 -d 24 -i input/random-n16384-d24-c16.txt -m 150 -t 0.00001 -s 8675309 -c &> local_testing/f_cuda2.txt
./bin/cuda_float -k 16 -d 16 -i input/random-n2048-d16-c16.txt -m 150 -t 0.00001 -s 8675309 -c &> local_testing/f_cuda3.txt


./bin/cuda_float -k 16 -d 32 -i input/random-n65536-d32-c16.txt -m 150 -t 0.00001 -s 8675309  &> local_testing/f_cuda_c.txt
./bin/cuda_float -k 16 -d 24 -i input/random-n16384-d24-c16.txt -m 150 -t 0.00001 -s 8675309  &> local_testing/f_cuda_c2.txt
./bin/cuda_float -k 16 -d 16 -i input/random-n2048-d16-c16.txt -m 150 -t 0.00001 -s 8675309  &> local_testing/f_cuda_c3.txt
