function [decompositions] = my_dwt(data,wavelet,level)
columns = {};
for i=1:level+2
    if i==1
        columns{i}='ORIG';
    elseif i==level+2
        columns{i}=['A',num2str(level)];
    else
        columns{i}=['D',num2str(i-1)];
    end  
end
%%----- Decompose the entire set
[C,L]=wavedec(data,level,wavelet);
%%% Extract approximation and detail coefficients
%%% Extract the approximation coefficients from C
cA=appcoef(C,L,wavelet,level);
%%% Extract the detail coefficients from C
cD = detcoef(C,L,linspace(1,level,level));
%%% Reconstruct the level approximation and level details
A=wrcoef('a',C,L,wavelet,level); %the approximation
for i=1:level
    eval(['D',num2str(i),'=','wrcoef(''d'',C,L,wavelet,i)',';']); %the details
end
%%% combine the details, appromaximation and original data into a single parameter
signals=zeros(length(data),level+2);
signals(:,level+2)=A;
signals(:,1)=data;
for i=2:level+1
    eval(['signals(:,i)','=','D',num2str(i-1),';']);
end
%%% save the decomposition results and the original data
decompositions = array2table(signals, 'VariableNames', columns);
end

