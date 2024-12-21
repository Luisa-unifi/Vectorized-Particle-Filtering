include "math.mc"

let model: () -> Float = lam.  
  let d = assume (Uniform 0. 2.) in
  let r = assume (Uniform 0. 1.) in
  let x = -1. in
  let y = 1. in
  let n0 = 0 in

  
  
   
   
  recursive let fact: Int -> Float -> Float -> Float -> Float -> Float =
    lam n0: Int. lam x: Float. lam y: Float. lam r: Float. lam d: Float. 
    let x1 = assume (Gaussian x d) in
    let y1 = assume (Gaussian y r) in
    let n1 = (addi n0 1) in

    let d2 =  subf x1 y1 in
    
    let answer = if (and (gtf 3. (subf x1 y1)) (gtf (subf x1 y1) -3. ) ) then 1. else 0. in
    observe true (Bernoulli answer);
    resample;
    
    let flag = if ((gti 1000 n1)) then 1. else 0. in
   
     if (or (and (gtf 0.1 (subf x y)) (gtf (subf x y) -0.1 ) ) (gti n1 1000)) then mulf r flag
     else fact n1 x1 y1 r d
    in 

  fact 0 x y r d
    
   
  
   
   
   
   

mexpr
model ()

