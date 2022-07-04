clear all
clc
load P_proteinA
load P_proteinB
load N_proteinA
load N_proteinB
num1=numel(P_proteinA);
result_Pa=[];
result_Pb=[];
result_Na=[];
result_Nb=[];
lambda=19;%9,10,11,13£¬14
 for i=1:num1
     result1=PAAC(P_proteinA{i},lambda);
     result_Pa=[result_Pa;result1];
     result1=[];
     result11=PAAC(P_proteinB{i},lambda);
     result_Pb=[result_Pb;result11];
     result11=[];
 end
  for i=1:num1
     result2=PAAC(proteinA{i},lambda);
     result_Na=[result_Na;result2];
     result2=[];
     result22=PAAC(proteinB{i},lambda);
     result_Nb=[result_Nb;result22];
     result22=[];
  end
  save PAAC_1_yeast.mat result_Pa result_Pb result_Na result_Nb
  csvwrite('S.core_PAAC_P112.csv', result_Pa)
  csvwrite('S.core_PAAC_P212.csv', result_Pb)
  csvwrite('S.core_PAAC_N112.csv', result_Na)
  csvwrite('S.core_PAAC_N212.csv', result_Nb)

