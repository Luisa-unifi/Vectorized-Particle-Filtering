"""
Python prototype implementation of the vectorized PF algorithm
described in "Parallelizable Feynman-Kac models for universal probabilistic 
programming", Section 6.

In this script we consider the Bounded retransmission model described in [1]:


The implementation is based on TensorFlow and autograph:
https://github.com/tensorflow/tensorflow/blob/master/
tensorflow/python/autograph/g3doc/reference/index.md.
"""

import time
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions





@tf.function (input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]*6) 
def brp(f,s,ber,SS,W,ZZ):  #with resampling
    print('Warm up')
    ones=1.0+ZZ
    ONE = 1.0+ZZ
    TWO = 2.0+ZZ
    FOUR = 4.0+ZZ
    #THREE = 3.0+ZZ
    def fbody(f,s,ber,SS,W):       
        
        EW=tf.math.exp(W)
        P=EW/tf.reduce_sum(EW)
        cum_dist = tf.math.cumsum(P[0])
        cum_dist /= cum_dist[-1]  # to account for floating point ehp2ors
        #unif_samp = tf.random.uniform((N,), 0, 1)
        unif_samp = tfd.Uniform(ZZ, ONE).sample()[0]#tf.random.uniform((N,), 0, 1)
        rs = tf.searchsorted(cum_dist, unif_samp)  # indices for resampling
        state_t = tf.concat([f,s,ber,SS],axis=0)    # state tensor  
        state_t = tf.gather(state_t,rs,axis=1) # resampled state tensor
        f,s,ber,SS= tuple([state_t[tf.newaxis,j] for j in range(4)])  # sliced state_t
        #W=ZZ
        #### END RESAMPLING 1 ##########
                 
    
        # MASKS DEFINITIONS
        mask0= (SS==0.0)
        mask1= (SS==1.0)& (s>0) & (f<=4)
        mask1b=(SS==1.0) & ((s==0) | (f>4))
        #mask2 =(SS==2.0)  
        mask2=(SS==2.0) & (ber==1.0)
        mask2b=(SS==2.0) & (ber!=1.0)
        #mask3 =(SS==3.0)
        mask4 =(SS==4.0)
        
        # STATE 0
        s=tf.where(mask0,100.0*ones,s) 
        f=tf.where(mask0,0.0*ones,f) 
        SS=tf.where(mask0,ONE,SS) 

        
        # STATE 1
        ber=tf.where(mask1,tf.cast(tfd.Bernoulli(probs=0.2*ones).sample(),tf.float32),ber)
        SS =tf.where(mask1,TWO,SS) 
        SS =tf.where(mask1b,FOUR,SS)
        
        # STATE 2
        f=tf.where(mask2,f+1,f)        
        f=tf.where(mask2b ,0.0,f)
        s=tf.where(mask2b ,s-1,s)        
        
        SS=tf.where(mask2,ONE,SS) 
        SS=tf.where(mask2b,ONE,SS)
        
        
        # STATE 4
        SS=tf.where(mask4,FOUR,SS)
        
        # W update        
        W=tf.where((SS==2.0),tf.math.log(tf.cast((ber!=1.0) | (s <= 80.0),tf.float32)),W)# mask2
                     
        return  (f,s,ber,SS,W) 

    def fcond(f,s,ber,SS,W):
        return True 
    
    f,s,ber,SS,W=tf.while_loop(fcond, fbody, (f,s,ber,SS,W), maximum_iterations=300,parallel_iterations=10,back_prop=True)  #maximum_iterations = fiter !
    return f,s,ber,SS,W



# warm up
N=1

ZZ=tf.zeros((1,N))
start_time=time.time()
res=brp(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ)
final_time=(time.time()-start_time)
print("TOTAL elapsed time  %s seconds -------        " % final_time)


#number f particles
N=10**5

ZZ=tf.zeros((1,N))
start_time=time.time()
res=brp(ZZ,ZZ,ZZ,ZZ,ZZ,ZZ)
final_time=(time.time()-start_time)
print("TOTAL elapsed time  %s seconds -------        " % final_time)


#weights
W=res[-1]
#final state
S=res[-2]

#output function
R1=res[1]
R=tf.where(R1 > 0, 1.0, 0.0)

# lower bound computation
EW=tf.math.exp(W)
P=EW/tf.reduce_sum(EW)
Rt=tf.where(S==4.0,R, 0.0)  
l_L = tf.reduce_sum(Rt*P)    

nil_n = 4.0  
M=2
# upper bound computation
Term = tf.where(S==nil_n,1.0, 0.0)      
alpha = 1/tf.reduce_sum(Term*P)
l_U = l_L*alpha + M*(alpha-1)