import numpy
import subprocess
import sys
import pdb

filename = '/home/siggy/PhD/gtzan/lists/audio_files.txt'

file_list = [l.strip() for l in open(filename,'r').readlines()]
pdb.set_trace()
#,'trim','0','30','pad','0','30']
cmd = ['sox',file_list[0],'-t','raw','-e','unsigned-integer','-L','-c','1','-b','16','-','pad','0','30.0','rate','22050.0','trim','0','30.0']
cmd = ' '.join(cmd)
#cmd = 'sox test.wav -t raw -e unsigned-integer -L -c 1 -b 16 - pad 0 30.0 rate 22050.0 trim 0 30.0'

#cmd = 'sox /home/siggy/PhD/gtzan/audio/blues/blues.00000.au -t raw -e unsigned-integer -L -c 1 -b 16 - pad 0 30.0 rate 22050.0 trim 0 30.0'

#cmd = cmd.split(' ')
#pdb.set_trace()
#subprocess.call(cmd,shell=True)
audio = numpy.fromstring(subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True).communicate()[0],dtype='uint16')
pdb.set_trace()
#p = 
#out,err = p.communicate()


	