include "math.mc"



let model: () -> Int = lam.

  let a0 = true in
  let b0 = true in
  
  let n0 = 0. in
  let c0 = a0 in
  let d0 = b0 in
  let n1 = (addf n0 1.) in

  recursive let fact: Int -> Bool -> Bool -> Bool -> Bool -> Int =
    lam n0: Int. lam a0: Bool. lam b0: Bool. lam c0: Bool. lam d0: Bool. 
    
    let a1 = assume (Bernoulli 0.5) in
    let b1 = assume (Bernoulli 0.5) in
    let answer = if or (or (and c0 a1) (and (not c0) (not a1))) (or (and d0 b1) (and (not d0) (not b1))) then 1. else 0. in
    observe true (Bernoulli answer);
    resample;
    let c1 = a1 in
    let d1 = b1 in
    let n1 = (addi n0 1) in
        
    let app = if (and (not a0) (not b0)) then 1 else 0 in        
    if (or (and (not a0) (not b0)) (gti n0 100)) then muli n0 app
     else fact n1 a1 b1 c1 d1
    in 

    
  fact 0 a0 b0 c0 d0
    




mexpr
model ()
