import time
import math
import json
#import ftfy 

def check_key(key,dic):
	exist=False
	for k in dic:
		if k == key:
			exist = True
			break
	return exist


def load_test_set():  # key is the UUID, values are the concepts
	testdic = dict()
	with open("en-es.5000-6500.txt", 'r') as f:
		line = f.readline()
		while line:
			wds  = line.strip().split(' ')
			orgw = str(wds[0].strip())
			trgw = str(wds[1].strip())
			#print (orgw,trgw)
			if check_key(orgw, testdic):
				testdic[orgw].add(trgw)
			else:
				testdic[orgw] = set()
				testdic[orgw]={trgw}
			line = f.readline()
	return testdic

def Load_List_of_most_similar():
	lmsFile = "UnSup-vectors-en.txt"
	lmsdic = {}
	with open(lmsFile, 'r') as smf:
		sline = smf.readline()
		while sline:
			sml  = sline.strip().split(' ')
			orgword = str(sml[0].strip().decode('unicode-escape'))
			#print(sline)
			trgwlist = (sml[1].replace('[','')).replace(']','').split(',')
			lmsdic[orgword] = trgwlist
			sline = smf.readline()
	return lmsdic

if __name__ == "__main__":

	found = 0.0
	t0 = time.time()
	testdic = load_test_set()
	lmsdic = Load_List_of_most_similar()
	t1 = time.time()
	print("loading the model takes %f seconds" % (t1 - t0))
	values= []
	pos = [0,0,0,0,0]
	for keyt , valuet in testdic.items():
		for keys, values in lmsdic.items():
			correct = False
			if keyt == keys:
				for val in list(valuet):
					if keyt==val.strip().decode('unicode-escape'):
						correct = True
						found = found + 1
						break
					for k in range (len(values)):
						if val.strip().decode('unicode-escape') == values[k].strip('\'').strip().decode('unicode-escape'):

							pos[k] += 1
							print ('%dst hit'%k)
							correct = True
					if correct:
						break		



	print('we had %d that have the same translation in our similar list'%(float(found)))
	#for j in range(0,20):
	print('we had %2f percent correct translation oft word of similar list' % ((float(sum(pos)+found)/len(lmsdic) )*100))
	print ('number of translation/List_of_most_similar is %d'% len(lmsdic) )





