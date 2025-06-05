
X=dlarray(rand([520, 704, 4],"single"), 'SSCB');


%%
A=extractdata(predict(net, X, Outputs='bn1'));
size(A)
max(A, [],"all")


%%
F=dlarray(zeros([260   352    64     1]));
%%
BN1=dlnetFeature.Layers(3);


%%


1==1;

B=extractdata(forward(BN1, F));
max(B, [], 'all')