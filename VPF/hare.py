"""
Python prototype implementation of the vectorized PF algorithm
described in "Parallelizable Feynman-Kac models for universal probabilistic 
programming" ([1]), Section 6.

In this script we consider the Hare and tortoise model described in [1].

The implementation is based on TensorFlow and autograph:
https://github.com/tensorflow/tensorflow/blob/master/
tensorflow/python/autograph/g3doc/reference/index.md.
"""


import time
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions




@tf.function (input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]*12) 
def hare(xx,yy,t0,tt,SS,W,ZERO,ONE,TWO,SEVEN,ZZ,ones):  #with resampling
    print('Warm up')
    def fbody(xx,yy,t0,tt,SS,W):
        ### RESAMPLING 1 ######
        EW=tf.math.exp(W)
        P=EW/tf.reduce_sum(EW)
        cum_dist = tf.math.cumsum(P[0])
        cum_dist /= cum_dist[-1]  # to account for floating point errors
        unif_samp = tfd.Uniform(ZZ, ONE).sample()[0]#tf.random.uniform((N,), 0, 1)
        rs = tf.searchsorted(cum_dist, unif_samp)  # indices for resampling
        state_t = tf.concat([xx,yy,t0,tt,SS],axis=0)    # state tensor  
        state_t = tf.gather(state_t,rs,axis=1) # resampled state tensor
        xx,yy,t0,tt,SS= tuple([state_t[tf.newaxis,j] for j in range(5)])  # sliced state_t
        #W=ZZ
        #### END RESAMPLING 1 ##########
        
    
        # MASKS DEFINITIONS
        mask0= (SS==0.0)
        mask1=(SS==1.0)& (xx<yy)
        mask2=(SS==1.0)& (xx>=yy)

        # STATE 0
        t0=tf.where(mask0,tfd.Uniform(0*ones, 10*ones).sample(),t0)  
        xx=tf.where(mask0,0*ones,xx)
        yy=tf.where(mask0,t0,yy)       

        tt=tf.where(mask0,0*ones,tt)
        SS=tf.where(mask0,ONE,SS) 

        SS=tf.where(mask2,SEVEN,SS) # FINAL STATE
        
        # STATE 1
        yy=tf.where(mask1,yy+1,yy)
        tt=tf.where(mask1,tt+1,tt)
        W=tf.where(SS==1.0,tf.math.log(tf.cast(tf.abs(xx-yy) <= 10.0,tf.float32)),W)# mask2

        SS=tf.where(mask2,SEVEN,SS)
     
        # STATE 2
        xx=tf.where(mask1&(tfd.Bernoulli(probs=2/5*ones).sample()==1),xx+tfd.Normal(loc=4*ones, scale=2*ones).sample(),xx)
        SS=tf.where(mask2,SEVEN,SS)

        # STATE 7
        W=tf.where(SS==SEVEN,W+tf.math.log(tf.cast(tt >= 20.0,tf.float32)),W)# mask2
                     
        return  (xx,yy,t0,tt,SS,W) 

    def fcond(xx,yy,t0,tt,ss,W):
        return True 
    
    xx,yy,t0,tt,SS,W=tf.while_loop(fcond, fbody, (xx,yy,t0,tt,SS,W), maximum_iterations=100,parallel_iterations=10,back_prop=True)  #maximum_iterations = fiter !
    return xx,yy,t0,tt,SS,W


# warm up
N=1

ZZ=tf.zeros((1,N))
start_time=time.time()
res=hare(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,1+ZZ,2+ZZ,7+ZZ,ZZ,1+ZZ)
final_time=(time.time()-start_time)
print("TOTAL elapsed time %s seconds -------        " % final_time)



N=10**6

ZZ=tf.zeros((1,N))
start_time=time.time()
res=hare(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,ZZ,1+ZZ,2+ZZ,7+ZZ,ZZ,1+ZZ)
final_time=(time.time()-start_time)
print("TOTAL elapsed time %s seconds -------        " % final_time)


#weights
W=res[-1]
#final state
S=res[-2]
#output function
R=res[0]   


# lower bound computation
EW=tf.math.exp(W)
P=EW/tf.reduce_sum(EW)
Rt=tf.where(S==7,R, 0.0)  
l_L = tf.reduce_sum(Rt*P)     


# effective sample size for EW
ess = tf.reduce_sum(EW)**2/(tf.reduce_sum(EW**2)) 


nil_n = 7.0  
M=2
# upper bound computation
Term = tf.where(S==nil_n,1.0, 0.0)      
alpha = 1/tf.reduce_sum(Term*P)
l_U = l_L*alpha + M*(alpha-1)      


#normalizing constant
norm = EW.numpy().sum()/len(EW.numpy()[0])