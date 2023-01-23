
import sys
sys.path.append("../AnotherIndicesTest/mdcgenpy/mdcgenpy")
import getopt
#sys.path.insert(0, '~/nestor/')
from clusters import ClusterGenerator
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from numpy import linalg as LA
from scipy.spatial import distance
import numpy as np
import pandas as pd
import os                           
import math                         
from sklearn import metrics
from numpy import linalg as LA
import statistics      
from sklearn.metrics.cluster import adjusted_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

def uncertainity_mean(X_train1,pred,means,covariances,weights,km,numberofdimens):
  X_train1["pred"]=pred
  meanss=means
  weiths=weights
  variances=[]
  covariancematrix=covariances
  for i in covariancematrix: #The diagonal of cov is the variance of each gmm
    diag=np.diag(i)
    variances.append(diag)
  variances=np.array(variances)
  countiterations=0
  unc=[]
  un=[]

  sep=[]
  for g in range(0,len(meanss)):  ##Frechet distance
    for t in range(0,len(meanss)):
      if (t==g):
        continue
      else:  
        sep.append(frechetDistance(meanss[t],meanss[g],covariancematrix[t],covariancematrix[g]))
  separation=np.array(sep)
  Dmin=np.array(separation).min()
  Dmax=np.array(separation).max()
  Sep=(Dmax/Dmin)*(1/separation.sum())

  for i in range(0,len(meanss)):  #Uncertainty  ##i run each cluster
    cweight=weiths[i]
    cmean=meanss[i]
    cvar=variances[i]
    lim_inf=cmean-(km*cvar)
    lim_sup=cmean+(km*cvar)
    unc=[]
    o=X_train1[X_train1["pred"]==i].iloc[:,:numberofdimens].copy()

    for j in range(len(o)): # j run each data 
      DM=distance.mahalanobis(np.array(o.iloc[j]), cmean, np.linalg.inv(covariancematrix[i]))
      varbool=(DM>=km)
      varunc=[]
      #print("varbool",varbool)
      if (varbool):
        varunc.append(2*km*DM)
      else:
        vs=((DM**2)+(km*DM)+(km**2)/2)
        varunc.append(vs)
    un.append(np.array(varunc)) 
  return np.sum(un)/Sep


def merguncertain (xtrain,y,Means,covariances,weights,numberofdimens):
  X_train11=xtrain
  X_train11["pred"]=y
  UNIndex=[]
  len(Means)
  Pairwise=np.ones((len(Means), len(Means)))
  labels=[]
  for i in range(len(Means)):
    labels.append(i)
    for j in range(len(Means)):
        if (i==j):
          continue
        Pairwise[j,i]=(frechetDistance(Means[i],Means[j],covariances[i],covariances[j]))
        

  while len(Pairwise) > 2 :
      UNIndex.append(uncertainity_mean(X_train11.iloc[:,:numberofdimens],X_train11["pred"],Means,covariances,weights,1,numberofdimens))

      Similar_clusters=np.unravel_index(Pairwise.argmin(),Pairwise.shape)  
      Similar_clusters_labels=( labels[Similar_clusters[0]],labels[Similar_clusters[1]])

      data_size_1=len(X_train11[X_train11["pred"]==Similar_clusters_labels[0]])
      data_size_2=len(X_train11[X_train11["pred"]==Similar_clusters_labels[1]])
      mean_1=Means[Similar_clusters[0]]
      mean_2=Means[Similar_clusters[1]] 
      new_mean = (data_size_1* mean_1 + data_size_2*mean_2)/(data_size_1 + data_size_2)
      Means[Similar_clusters[0]]=new_mean

      X_train11["pred"]=X_train11["pred"].replace(Similar_clusters_labels[1], Similar_clusters_labels[0])
      #print(X_train11["pred"].unique())
      NewCovariance=X_train11[X_train11["pred"]==Similar_clusters_labels[0]].iloc[:,:numberofdimens].cov()
      covariances[Similar_clusters[0]]=NewCovariance
      weights[Similar_clusters[0]]=weights[Similar_clusters[0]]+weights[Similar_clusters[1]]

      updpairwisecolum=[]
      for j in range(0,len(Means)):
          updpairwisecolum.append((frechetDistance(Means[Similar_clusters[0]],Means[j],covariances[Similar_clusters[0]],covariances[j])))
      #print(updpairwisecolum)    
      Pairwise[:,Similar_clusters[0]]=updpairwisecolum
      Pairwise[Similar_clusters[0],:]=updpairwisecolum

      Means= np.delete(Means, (Similar_clusters[1]), axis=0)
      weights= np.delete(weights, (Similar_clusters[1]), axis=0)
      covariances=np.delete(covariances, (Similar_clusters[1]), axis=0)
      labels.remove(Similar_clusters_labels[1])
      Pairwise= np.delete(Pairwise, (Similar_clusters[1]), axis=0)
      Pairwise= np.delete(Pairwise, (Similar_clusters[1]), axis=1)
      np.fill_diagonal(Pairwise, 1)
  return UNIndex

def frechetDistance(u1,u2,E1,E2): 
  return (LA.norm(np.absolute(u1-u2)**2))+np.trace(E1+E2-(2*(E1*E2)**(0.5)))

def genData(n_samples,n_featuress, n_components,cl_std,randomstate):
  X, y_true = make_blobs(
        n_samples, n_features, centers=n_components, cluster_std=cl_std, random_state=randomstate
    )
  X = X[:, ::-1]
  dataFrame=pd.DataFrame(X)
  dataFrame["y"]=y_true
  return dataFrame

def iterateindex(inf,sup):
  cpred=[]
  admis=[]
  for index, row in car.iloc[inf:sup,3:12].iterrows():
    print(index)
    cluster_gen=ClusterGenerator(seed=1, n_samples=row.samples, n_feats=row.n_feats, k=row.k_real, 
                      distributions=row.distributions, mv=False, corr=0., compactness_factor=row.compactness_factor, alpha_n=row.alpha_n,
                      scale=True, outliers=row.outliers, rotate=False, add_noise=int(row.add_noise),  ki_coeff=3.)
    data = cluster_gen.generate_data()
    dataFrame= pd.DataFrame(data[0]).round(3)
    dataFrame['y']=data[1]
    kgmm=row.k_real+10 ##int(row.k_GMM.split(":")[1])

    X_train111=dataFrame.iloc[:,:row.n_feats].copy()
    GM = GaussianMixture(n_components=kgmm, covariance_type="full",random_state=1).fit(X_train111)
    X_train111["pred"]=GM.predict(X_train111)
    Means=GM.means_
    covariances=GM.covariances_
    weights=GM.weights_
    xtrain=X_train111.iloc[:,:row.n_feats]
    y=X_train111["pred"]
        
    index=merguncertain (xtrain,y,Means,covariances,weights,row.n_feats)
    index.reverse()
    change=pd.DataFrame(index,columns=["a"])
    recomended=change["a"].argmin()+2
    GMM_test= GaussianMixture(n_components=recomended, covariance_type="full",random_state=1).fit(X_train111.iloc[:,:row.n_feats])
    predy=GMM_test.predict(X_train111.iloc[:,:row.n_feats])
    admis.append(adjusted_mutual_info_score(dataFrame['y'], predy))
    cpred.append(recomended)
  return admis,cpred


def arguments(argv):
    

    arg_filepath=""
    arg_savepath=""

    arg_start=""
    arg_end=""

    arg_help = "{0} -f <filepath> -sp <savepath>  -c <start> -e <end>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hi:f:s:c:e:", ["help", "filepath=", 
        "savepath=",  "start=", "end="])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-f", "--filepath"):
            arg_filepath = arg
            print('arggg',arg)
        elif opt in ("-s", "--savepath"):
            arg_savepath = arg
        elif opt in ("-c", "--start"):
            arg_start = arg
        elif opt in ("-e", "--end"):
            arg_end = arg


    return arg_filepath, arg_savepath, int(arg_start), int(arg_end)


if __name__ == "__main__":
  filepath,savepath,start,end =arguments(sys.argv)
  print('**FILEPATH**',filepath)
  car=pd.read_excel(filepath)  

  for i in range(start,end,10):
    admis, cpred =iterateindex(i,i+10)
    with open(savepath+"admsco_"+str(i)+"_"+str(i+10)+".txt", 'w') as f:
        for s in admis:
            f.write(str(s) + '\n')
    with open(savepath+"clusterspred_"+str(i)+"_"+str(i+10)+".txt", 'w') as f:
        for s in cpred:
            f.write(str(s) + '\n')