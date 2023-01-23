import sys

sys.path.insert(0, '/home/cisaza/Downloads/mdcgenpy/mdcgenpy')
from clusters import ClusterGenerator
sys.path.insert(0, '/home/cisaza/Downloads/mdcgenpy/mdcgenpy/Clusters-Features')
from ClustersFeatures import *
from clusters import ClusterGenerator
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import adjusted_mutual_info_score
import os                           
import math                         
import numpy as np                  
from scipy import signal, stats     
from scipy.fft import fftshift          
import statistics                   
from sklearn import metrics
from numpy import linalg as LA
from scipy.spatial import distance                     

def iterateindex(inf,sup):
  cpred=[]
  admis=[]

  ncreal=[]
  ncpred_slope=[]
  ncpred_min=[]
  sil=[]
  silluette=[]
  bh=[]
  rtl=[]
  sd=[]
  db=[]
  xb=[]
  ch=[]
  dnn=[]
  mr=[]
  pb=[]
  rt=[]
  wg=[]
  dr=[]
  c=[]

  for index, row in car.iloc[inf:sup,3:12].iterrows():
    print(index)
    cluster_gen=ClusterGenerator(seed=1, n_samples=row.samples, n_feats=row.n_feats, k=row.k_real, 
                      distributions=row.distributions, mv=False, corr=0., compactness_factor=row.compactness_factor, alpha_n=row.alpha_n,
                      scale=True, outliers=row.outliers, rotate=False, add_noise=int(row.add_noise),  ki_coeff=3.)
    data = cluster_gen.generate_data()
    dataFrame= pd.DataFrame(data[0]).round(3)
    dataFrame['y']=data[1]
    kgmm=row.k_real+20

    varuncer=[]
    separation=[]
    sil=[]
    g1=[]
    g2=[]
    g3=[]
    g4=[]
    g5=[]

    g6=[]
    g7=[]
    g8=[]
    g9=[]
    g10=[]
    g11=[]
    g12=[]
    g13=[]
    X_train11=dataFrame.iloc[:,:row.n_feats].copy()
    for l in range(2,kgmm):
      rand=1
      
      #unce, sep,pred=uncertainity_mean(X_train11,n_components=l)
      GM = GaussianMixture(n_components=l, random_state=rand, covariance_type="full").fit(X_train11)
      pred=GM.predict(X_train11)
      sil.append(metrics.silhouette_score(X_train11, pred, metric='euclidean'))
      X_train11["target"]=pred

      CC=ClustersCharacteristics(X_train11,label_target="target")
      g1.append(CC.score_index_ball_hall())
      g2.append(CC.score_index_ratkowsky_lance())
      g3.append(CC.score_index_SD())
      g5.append(CC.score_index_xie_beni())
      g6.append(CC.score_index_calinski_harabasz())
      g7.append(CC.score_index_dunn())
      g8.append(CC.score_index_mclain_rao())
      g9.append(CC.score_index_point_biserial())
      g10.append(CC.score_index_ray_turi())
      g11.append(CC.score_index_wemmert_gancarski())
      g13.append(CC.score_index_c())
    
      try:
          g4.append(CC.score_index_davies_bouldin())
      except:
          print("An exception occurred")
          g4.append(0)
    silluette.append(np.array(sil).argmax()+2)
    bh.append(np.diff(np.array(g1)).argmax()+2)
    rtl.append(np.array(g2).argmax()+2)
    sd.append(np.array(g3).argmin()+2)
    db.append(np.array(g4).argmin()+2)
    xb.append(np.array(g5).argmin()+2)
    ch.append(np.array(g6).argmax()+2)
    dnn.append(np.array(g7).argmax()+2)
    mr.append(np.array(g8).argmin()+2)
    pb.append(np.array(g9).argmax()+2)
    rt.append(np.array(g10).argmin()+2)
    wg.append(np.array(g11).argmax()+2)
    c.append(np.array(g13).argmin()+2)

    ncrecomended=[silluette,bh,rtl,sd,db,xb,ch,dnn,mr,pb,rt,wg,c]
    print('lenrecomended',len(ncrecomended))
    adm=[]
    for i in range(len(ncrecomended)):
      GMM_test= GaussianMixture(n_components=ncrecomended[i][0], covariance_type="full",random_state=1).fit(X_train11.iloc[:,:row.n_feats])
      predy=GMM_test.predict(X_train11.iloc[:,:row.n_feats])
      adm.append(adjusted_mutual_info_score(dataFrame['y'], predy))
    print('adm',adm)
    cpred.append(ncrecomended)  
    admis.append(adm)
    print('admis',admis)
  return admis,cpred


def runCIVIS(inf,sup,path,pathfile):
  
    for i in range(inf,sup,10):
      admis, cpred =iterateindex(i,i+10)
      with open(path+"admsco_"+str(i)+"_"+str(i+10)+".txt", 'w') as f:
          for s in admis:
              f.write(str(s) + '\n')
      with open(path+"clusterspred_"+str(i)+"_"+str(i+10)+".txt", 'w') as f:
          for s in cpred:
              f.write(str(s) + '\n')
inf=0
sup=500
pathfile="/home/cisaza/Downloads/info_generateddata.xlsx"
car=pd.read_excel(pathfile)
path="/home/cisaza/Downloads/normal/"
runCIVIS(inf,sup,path,pathfile)            
