rng(12);
A = rand(10,3);
b = rand(10,1);rank(A);  % note A has full rank (rank = 3)

x = A \ b;
[Q,R] = qr(A); 

y = Q \ b;
z = R \ y;

R(3,3) = 0;
As = Q*R; % make As a matrix with linearly dependent columns
rank(As); % note As is rank-deficient (rank = 2)

[C,D] = qr(As);

r = Q \ b
s = R \ y