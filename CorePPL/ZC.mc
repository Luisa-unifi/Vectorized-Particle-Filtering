include "math.mc"

let model: () -> Float = lam. 
  let prob = assume (Uniform 0. 1.) in
  let z = 0 in
  let est= 0 in
  let c= 0 in
  let n0 = 0 in
  
   
  recursive let fact: Int -> Int-> Int  -> Int -> Float -> Float =
    lam n0: Int. lam c: Int. lam z: Int. lam est: Int. lam prob: Float.  
    

    let n1 = (addi n0 1) in    
    let flag = if ((gti 100 n0)) then 1. else 0. in  

    let app = or (gti c 99) (gti z 1) in
    if or (or (gti est 0 )  (gti n0 99)) app then   

        mulf prob flag
    else   
       let flag1 = assume (Bernoulli prob) in
       let est = if (and (not flag1) (gti z 0 )) then 1 else est in
       let z = if (gti z 0 ) then 0 else z in
       

       let flag2= assume (Bernoulli 0.5) in
       let c = if (and (flag2) (gti 1 z ))  then (addi c 1)  else c in
       let c = if (and (not flag2) (gti 1 z ))  then 0 else c in
       let z = if (and (not flag2) (gti 1 z ))  then 1 else z in
       
  (if and (not flag2) (gti 1 z )  then
       let answer = if (gti c 20) then 1. else 0. in
       observe true (Bernoulli answer); resample
     else ());


        fact n1 c z est prob 
      in 
               
       

  fact n0 c z est prob 
    
   

mexpr
model ()




   
