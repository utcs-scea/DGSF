./bin/thrust_impl -k 16 -d 32 -i input/random-n65536-d32-c16.txt -m 150 -t 0.00001 -s 8675309 -c &> local_testing/impl_thrust.txt
./bin/thrust_impl -k 16 -d 24 -i input/random-n16384-d24-c16.txt -m 150 -t 0.00001 -s 8675309 -c &> local_testing/impl_thrust2.txt
./bin/thrust_impl -k 16 -d 16 -i input/random-n2048-d16-c16.txt -m 150 -t 0.00001 -s 8675309 -c &> local_testing/impl_thrust3.txt


./bin/thrust_impl -k 16 -d 32 -i input/random-n65536-d32-c16.txt -m 150 -t 0.00001 -s 8675309  &> local_testing/impl_thrust_c.txt
./bin/thrust_impl -k 16 -d 24 -i input/random-n16384-d24-c16.txt -m 150 -t 0.00001 -s 8675309  &> local_testing/impl_thrust_c2.txt
./bin/thrust_impl -k 16 -d 16 -i input/random-n2048-d16-c16.txt -m 150 -t 0.00001 -s 8675309  &> local_testing/impl_thrust_c3.txt
