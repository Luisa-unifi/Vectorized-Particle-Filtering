include "math.mc"

let model: () -> Float = lam. 
  let r = assume (Uniform 0. 7.) in
  let z = 0.5 in
  let c= 1. in
  let n0 = 0 in
  
   
  recursive let fact: Int -> Float-> Float  -> Float -> Float =
    lam n0: Int. lam c: Float. lam z: Float. lam r: Float.  
    

    let n1 = (addi n0 1) in 
    let c0 = c in       

    if (gti n0 100) then   
        mulf c 1.0
    else   
       let flag = assume (Bernoulli z) in     
       let c = assume (Gaussian c (mulf 2.0 r)) in
       
        (if (flag) then   
        let answer = if (ltf (absf (subf c c0)) 2. )  then 1. else 0. in   
       observe true (Bernoulli answer); resample
     else ());
 
       
       fact n1 c z r
      in 
               
       

  fact n0 c z r
    
   

mexpr
model ()




   
