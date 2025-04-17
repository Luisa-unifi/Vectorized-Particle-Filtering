"""
Python prototype implementation of the vectorized PF algorithm
described in "Parallelizable Feynman-Kac models for universal probabilistic 
programming", Section 6.

In this script we consider the ZeroConf (ZC):


The implementation is based on TensorFlow and autograph:
https://github.com/tensorflow/tensorflow/blob/master/
tensorflow/python/autograph/g3doc/reference/index.md.
"""

import time
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions



@tf.function (input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]*12) 
def running_example(xx,yy,rr,cc,est,SS,W,ZERO,ONE,SEVEN,ZZ,ones):  #with resampling
    print('Warm up')
    def fbody(xx,yy,rr,cc,est,SS,W):
        
        EW=tf.math.exp(W)
        P=EW/tf.reduce_sum(EW)
        cum_dist = tf.math.cumsum(P[0])
        cum_dist /= cum_dist[-1] 
        unif_samp =  tfd.Uniform(low=ZZ).sample()[0] #tf.random.uniform((N,), 0, 1)
        rs = tf.searchsorted(cum_dist, unif_samp)  # indices for resampling
        state_t = tf.concat([xx,yy,rr,cc,est,SS],axis=0)    
        state_t = tf.gather(state_t,rs,axis=1) # resampled state tensor
        xx,yy,rr,cc,est,SS= tuple([state_t[tf.newaxis,j] for j in range(6)])  # sliced state_t
                
    
        # MASKS DEFINITIONS
        mask0= (SS==0.0)
        mask1=(SS==1.0)& (est<1.0)
        mask2=(SS==1.0)& (est>0.0)


        # STATE 0
        rr=tf.where(mask0,tfd.Uniform(0.*ones, 1.*ones).sample(),rr)   #0.99999*ones


        SS=tf.where(mask0,ONE,SS) 
        SS=tf.where(mask2,SEVEN,SS) 

        
        # STATE 1
        flag=tf.cast(tfd.Bernoulli(probs=rr*ones).sample(),tf.float32)
        est=tf.where(mask1&(yy>0.0)&(flag<1),1.0,est)
        yy=tf.where(mask1&(yy>0.0),0.0,yy)
        
        
        flag2=tf.cast(tfd.Bernoulli(probs=0.99).sample(),tf.float32)
        xx=tf.where(mask1&(yy<1.0)&(flag2>0),xx+1,xx)
        xx=tf.where(mask1&(yy<1.0)&(flag2<1),0.,xx)
        yy=tf.where(mask1&(yy<1.0)&(flag2<1),1.0,yy)


        W=tf.where(mask1&(yy<1.0)&(flag2<1),tf.math.log(tf.cast(xx > 20,tf.float32)),W)
        SS=tf.where(mask1,ONE,SS)


                     
        return  (xx,yy,rr,cc,est,SS,W) 

    def fcond(xx,yy,rr,cc,est,ss,W):
        return tf.logical_not(tf.reduce_all(SS >= 7)) 
    
    xx,yy,rr,cc,est,SS,W=tf.while_loop(fcond, fbody, (xx,yy,rr,cc,est,SS,W), maximum_iterations=100,parallel_iterations=10,back_prop=True)  #maximum_iterations = fiter !
    return xx,yy,rr,cc,est,SS,W




#number of particles
N=1

ZZ=tf.zeros((1,N))
start_time=time.time()
res=running_example(ZZ,1+ZZ,ZZ,ZZ+1,ZZ,ZZ,ZZ,ZZ,1+ZZ,7+ZZ,ZZ,1+ZZ)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1 elem  %s seconds -------        " % final_time)


#number of particles
N=10**5

ZZ=tf.zeros((1,N))
start_time=time.time()
res=running_example(ZZ,1+ZZ,ZZ,ZZ+1,ZZ,ZZ,ZZ,ZZ,1+ZZ,7+ZZ,ZZ,1+ZZ)
final_time=(time.time()-start_time)
print("TOTAL elapsed time 1M elems  %s seconds -------        " % final_time)


#weights
W=res[-1]
#final state
S=res[-2]
#output function
R=res[2]   


# lower bound computation
EW=tf.math.exp(W)
P=EW/tf.reduce_sum(EW)
Rt=tf.where(S==7.0,R, 0.0)  
l_L = tf.reduce_sum(Rt*P)     

# effective sample size for EW
ess = tf.reduce_sum(EW)**2/(tf.reduce_sum(EW**2)) 


