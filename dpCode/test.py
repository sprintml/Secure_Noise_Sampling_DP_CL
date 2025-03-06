import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_parties',  action='store', dest='n', type = int, required = True)
parser.add_argument('-s', '--start_party',  action='store', dest='s', type = int, required = True)
parser.add_argument('-e', '--end_party',  action='store', dest='e', type = int, required = True)
parser.add_argument('-p', '--num_processes',  action='store', dest='p', type = int, required = False)
args = parser.parse_args()

def test(n):
    s = ""
    for i in range(args.s, args.e):
        s += "./build/bin/test_vec_gen " + ( i + 1).__str__() + " 12345 " + n.__str__() + " ip_file.txt 14"
        if i != args.e-1: 
            s += " & "
    print(s)
    return s

sleep_time_batch = 70
#test(args.n)
if args.p:
    s = ""
    sleep_time = 0
    for i in range(args.p):
        s += "sleep " + (sleep_time * sleep_time_batch).__str__() + "; " + test(args.n)
        sleep_time += 1
        if i != args.p-1:
            s += " & "
        
    print(s)
    subprocess.Popen(s, shell="True")
else:
    subprocess.Popen(test(args.n), shell="True")