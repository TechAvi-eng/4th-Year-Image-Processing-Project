

for j = [0.001:0.01:0.3]

    for i = 1:569
numpred(i) = nnz(allpsocres{i}>j);
%numpred(i) = size(allpboxes{i},1);
numac(i) = size(allabbox{i},1);



    end

ids = numac<123123123;

numac = numac(ids);
length(numac)
numpred = numpred(ids);

j
%mean(abs(numac-numpred)./(numpred+numac))*100
sqrt(mean((numac-numpred).^2))

end