import numpy as np
import pywt
import matplotlib.pyplot as plt
#import copy
from PIL import Image

class DWT(object):
	def __init__(self, coeff_r, coeff_g, coeff_b, wavelet): #initialisiert ein object DWT 
		self.LL_dense = [coeff_r[0], coeff_g[0], coeff_b[0]]
		self.LH_sparse = [SparseMatrix(coeff_r[1][0]), SparseMatrix(coeff_g[1][0]), SparseMatrix(coeff_b[1][0])]
		self.HL_sparse = [SparseMatrix(coeff_r[1][1]), SparseMatrix(coeff_g[1][1]), SparseMatrix(coeff_b[1][1])]	
		self.HH_sparse = [SparseMatrix(coeff_r[1][2]), SparseMatrix(coeff_g[1][2]), SparseMatrix(coeff_b[1][2])]
		self.wavelet = wavelet
	def invert(self):  #für a aus DWT classe: a.invert()
		r_invert = pywt.idwt2((self.LL_dense[0], (self.LH_sparse[0].get_dense(), self.LH_sparse[0].get_dense(), self.HH_sparse[0].get_dense())), self.wavelet)
		g_invert = pywt.idwt2((self.LL_dense[1], (self.LH_sparse[1].get_dense(), self.LH_sparse[1].get_dense(), self.HH_sparse[1].get_dense())), self.wavelet)
		b_invert = pywt.idwt2((self.LL_dense[2], (self.LH_sparse[2].get_dense(), self.LH_sparse[2].get_dense(), self.HH_sparse[2].get_dense())), self.wavelet)
		return np.stack((r_invert, g_invert, b_invert), axis=2)
	def plot_DWT(self, colour): #colour is 'r', 'g', 'b' or 'all'
		if colour == 'r':
			colour_number = 0
		if colour == 'g':
			colour_number = 1
		if colour == 'b':
			colour_number = 2
		#colour all 
		titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
		fig = plt.figure(figsize=(96, 24))       
		for i, a in enumerate([self.LL_dense[colour_number], self.LH_sparse[colour_number].get_dense(), self.HL_sparse[colour_number].get_dense(), self.HH_sparse[colour_number].get_dense()]):
		    ax = fig.add_subplot(1, 4, i + 1)
		    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
		    ax.set_title(titles[i], fontsize=10)
		    ax.set_xticks([])
		    ax.set_yticks([])
	def get_DWT(self):
		return ((self.LL_dense[0], (self.LH_sparse[0].get_dense(), self.LH_sparse[0].get_dense(), self.HH_sparse[0].get_dense())), (self.LL_dense[1], (self.LH_sparse[1].get_dense(), self.LH_sparse[1].get_dense(), self.HH_sparse[1].get_dense())), (self.LL_dense[2], (self.LH_sparse[2].get_dense(), self.LH_sparse[2].get_dense(), self.HH_sparse[2].get_dense())), self.wavelet)
		#kann idwt von output 1 2 3 r g b aufrufen
	def storage_size(self): #nach sparsify aufrufen
		#anzahl variablen 
		var_ll=self.LL_dense[0].shape[0]*self.LL_dense[0].shape[1]+self.LL_dense[1].shape[0]*self.LL_dense[1].shape[1]+self.LL_dense[2].shape[0]*self.LL_dense[2].shape[1] 
		var_lh=self.LH_sparse[0].stored_elements()+self.LH_sparse[1].stored_elements()+self.LH_sparse[2].stored_elements()
		var_hl=self.HL_sparse[0].stored_elements()+self.HL_sparse[1].stored_elements()+self.HL_sparse[2].stored_elements()
		var_hh=self.HH_sparse[0].stored_elements()+self.HH_sparse[1].stored_elements()+self.HH_sparse[2].stored_elements()
		return var_ll+var_lh+var_hl+var_hh
	def sparsify_perc(self, perc): #nimmt kopiert und gibt es anders zurüc (lässt altes so)
		#output=copy.deepcopy(self) #deepcopy nimmt jedes object an
		#nehmen matrix ,machen daraus dense?!?!?! dann löschen wir entries, dann sparsifien wirund das alles in dieser function 
		output=self
		for i in range(3):
			output.LH_sparse[i]=SparseMatrix(threshold_percentage(output.LH_sparse[i].get_dense(), perc))
			output.HL_sparse[i]=SparseMatrix(threshold_percentage(output.HL_sparse[i].get_dense(), perc))
			output.HH_sparse[i]=SparseMatrix(threshold_percentage(output.HH_sparse[i].get_dense(), perc))
		return output
	def sparsify_abs(self, val):
		#print('start')
		#output=copy.deepcopy(self)
		#print('enddd')
		output=self
		for i in range(3):
			output.LH_sparse[i]=SparseMatrix(threshold_absolute(output.LH_sparse[i].get_dense(), val))
			output.HL_sparse[i]=SparseMatrix(threshold_absolute(output.HL_sparse[i].get_dense(), val))
			output.HH_sparse[i]=SparseMatrix(threshold_absolute(output.HH_sparse[i].get_dense(), val))
		return output
class SparseMatrix(object):  
    def __init__(self, dwt_comp):  
        self.size = dwt_comp.shape 
        self.indices = []
        self.values = []
        for i in range(dwt_comp.shape[0]):
            for j in range(dwt_comp.shape[1]):
                if dwt_comp[i][j] != 0:
                    self.indices.append((i,j))
                    self.values.append(dwt_comp[i][j])
    def get_dense(self):  
        output = np.zeros(self.size)
        for i in range(len(self.indices)):
            output[self.indices[i][0],self.indices[i][1]]=self.values[i]                    
        return output
    def stored_elements(self): 
    	return len(self.values)*3

def threshold_percentage(mat, perc): #we keep perc % of the amount of our values (the largest prc percent)
 	#perc shoult be [0,1]
 	#wir löschen nicht wirklich die prozent weil es sein kann das der wert dort oft vorkommt
 	amount = mat.shape[0]*mat.shape[1] 
 	in_order=np.sort(mat.flatten())
 	benchmark = round(amount*(1-perc))
 	return pywt.threshold(mat, in_order[benchmark], mode= 'greater', substitute=0)

def threshold_absolute(mat, val): #the values below val will be set to 0
 	return pywt.threshold(mat, val, mode='greater', substitute=0)

def create(img, wavelet='bior1.3'): #zb bior haar (schreibweise)
	r=img[:,:,0]
	g=img[:,:,1]
	b=img[:,:,2]
	return DWT(pywt.dwt2(r, wavelet), pywt.dwt2(g, wavelet), pywt.dwt2(b, wavelet), wavelet)
 	#rück dwt obk, input bild als array
def get_array(img_str):
	return np.asarray(Image.open(f"data/{img_str}.jpg"))



#welche idee? sparzifizieren