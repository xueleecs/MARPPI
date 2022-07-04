clear all
clc
load P_protein_a
load P_protein_b
load('abc.mat');
mmi_map = abc;

feature_MMI = [];
A_feature=[];
B_feature=[];
%P_A
protein_A_sequence_f_1=P_protein_a;
protein_B_sequence_f_1=P_protein_b;
for i=1:size(protein_A_sequence_f_1)
    SEQ=protein_A_sequence_f_1(i);
	F = MMI(SEQ,mmi_map);
    A_feature(i,:)=F;
	kd = mod(i,100);
	if kd==0
		prin = i;
		prin
	end
end


%P_B
for i=1:size(protein_B_sequence_f_1)
    SEQ=protein_B_sequence_f_1(i);
	F = MMI(SEQ,mmi_map);
    B_feature(i,:)=F;
	kd = mod(i,100);
	if kd==0
		prin = i;
		prin
	end
end

feature_MMI = [A_feature,B_feature];


