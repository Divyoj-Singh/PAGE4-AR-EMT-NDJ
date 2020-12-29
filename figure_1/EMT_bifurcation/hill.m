% hill function
function H = hill(X,X0,n)
% H
H = (X^n)/(X0+X^n);
end