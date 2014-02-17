import numpy
import subprocess
import sys
import pdb

filename = '/home/siggy/PhD/gtzan/lists/audio_files.txt'

file_list = [l.strip() for l in open(filename,'r').readlines()]

cmd = ['sox',file_list[0],'-t','raw','-e','unsigned-integer','-L','-c','1','-b','16']#,'trim','0','30','pad','0','30']
cmd = 'sox test.wav -t raw -e unsigned-integer -L -c 1 -b 16 - pad 0 30.0 rate 22050.0 trim 0 30.0'
stderr = subprocess.PIPE

#subprocess.call(cmd,shell=True)
audio = numpy.fromstring(subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True).communicate()[0],dtype='uint16')
#p = 
#out,err = p.communicate()
pdb.set_trace()

