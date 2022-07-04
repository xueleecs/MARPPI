function feature=MMI(protein,abc)

MMI_ab=zeros(1,28);
groups_no = 7;
protein=cell2mat(protein);
%%%%%%%%%%%%%%%%%%%% amino acid grouping  
protein=regexprep(protein,'X',''); %%%% delete unknown proteins
protein=regexprep(protein,'A|G|V','1');
protein=regexprep(protein,'C','2');
protein=regexprep(protein,'D|E','3');
protein=regexprep(protein,'I|L|F|P','4');
protein=regexprep(protein,'H|N|Q|W','5');
protein=regexprep(protein,'R|K','6');
protein=regexprep(protein,'Y|M|T|S','7');




L = length(protein);
protein_groupings_index='1234567';
protein_number_index=1:1:7;


%%%% composition descriptors
CD = zeros(1,groups_no);
F_A = zeros(1,groups_no);
for i=1:groups_no
	num_a = length(regexp(protein,protein_groupings_index(i)));
    CD(i)= (num_a+1)/(L+1);
	F_A(i)= num_a/L;
end


TD = zeros(groups_no,groups_no);
temp = zeros(groups_no,groups_no);
Matrix =zeros(groups_no,groups_no);
f_ab = zeros(groups_no,groups_no);

for i=1:L-1
    m=protein_number_index(:,findstr(protein_groupings_index,protein(i)));
    n=protein_number_index(:,findstr(protein_groupings_index,protein(i+1)));
    temp(m,n)=temp(m,n)+1;    
end


for i=1:groups_no-1
    for j=i+1:groups_no
        TD(i,j)=(temp(i,j)+temp(j,i));
		TD(j,i)=TD(i,j);
    end
end

for i=1:groups_no
    TD(i,i)=temp(i,i);
end 

%MI
for i=1:groups_no
    for j=i:groups_no
        fen_zi = (TD(i,j)+1)/(L-1+1);
		fen_mu = CD(i)*CD(j);
		f_ab(i,j) = (TD(i,j)+1)/(L-1+1);
		f_ab(j,i) = f_ab(i,j);
		value_x = fen_zi * log(fen_zi/fen_mu);
		Matrix(i,j) = value_x;
    end
end


k=1;
for i=1:groups_no
    for j=i:groups_no
        MMI_ab(k)=Matrix(i,j);
        k=k+1;
    end
end


%MMI

n_MMI = size(abc,1);
n_abc = zeros(n_MMI,1);
f_abc = zeros(n_MMI,1);
MMI_abc = zeros(n_MMI,1);

His_abc = [];
His_reasch = [];

for i = 1:n_MMI
	His_abc = [];
	ll_abc = abc(i,:);
	h0=length(find(ll_abc==1));
	h1=length(find(ll_abc==2));
	h2=length(find(ll_abc==3));
	h3=length(find(ll_abc==4));
	h4=length(find(ll_abc==5));
	h5=length(find(ll_abc==6));
	h6=length(find(ll_abc==7));
	His_abc = [h0;h1;h2;h3;h4;h5;h6];
	for j=1:L-2
		His_reasch = [];
		a=protein_number_index(:,findstr(protein_groupings_index,protein(j)));
		b=protein_number_index(:,findstr(protein_groupings_index,protein(j+1)));
		c=protein_number_index(:,findstr(protein_groupings_index,protein(j+2))); 
		nn_abc = [a;b;c];
		h_s0=length(find(nn_abc==1));
		h_s1=length(find(nn_abc==2));
		h_s2=length(find(nn_abc==3));
		h_s3=length(find(nn_abc==4));
		h_s4=length(find(nn_abc==5));
		h_s5=length(find(nn_abc==6));
		h_s6=length(find(nn_abc==7));
		His_reasch = [h_s0;h_s1;h_s2;h_s3;h_s4;h_s5;h_s6];
		ret = sum(His_abc==His_reasch);
		if ret==7
			n_abc(i,1) = n_abc(i,1) + 1;
		end
	end
	p_abc = (n_abc(i,1)+1)/(L-2+1); 
	f_abc(i,1) = p_abc;
	a_1 = ll_abc(1,1);a_2 = ll_abc(1,2);a_3 = ll_abc(1,3);
	%%
	p_ab = f_ab(a_1,a_2);
	p_ac = f_ab(a_1,a_3);
	p_bc = f_ab(a_2,a_3);
	%%
	p_a = CD(a_1);
	p_b = CD(a_2);
	p_c = CD(a_3);
	value_mmi = p_ab*log(p_ab/(p_a*p_b)) - (-1*(p_ac/p_c)*log(p_ac/p_c) + (p_abc/p_bc)*log(p_abc/p_bc));
	MMI_abc(i,1) = value_mmi;

end

MMI_ab = MMI_ab';
F_A = F_A';
feature = [MMI_abc;MMI_ab;F_A];



feature(find(isnan(feature)))=0;
feature(find(isinf(feature)))=0;