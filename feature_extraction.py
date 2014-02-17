import numpy
import subprocess
import sys
import pdb

def read_wav(filename):
	bits_per_sample = '16'
	cmd = ['sox',file_list[0],'-t','raw','-e','unsigned-integer','-L','-c','1','-b',bits_per_sample,'-','pad','0','30.0','rate','22050.0','trim','0','30.0']
	cmd = ' '.join(cmd)
	raw_audio = numpy.fromstring(subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True).communicate()[0],dtype='uint16')
	max_amp = 2.**(int(bits_per_sample)-1)
	pdb.set_trace()
	raw_audio = (raw_audio- max_amp)/max_amp
	return raw_audio

filename = '/home/siggy/PhD/gtzan/lists/audio_files.txt'

file_list = [l.strip() for l in open(filename,'r').readlines()]

data = read_wav(file_list[0])

