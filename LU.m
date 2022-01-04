
n = 100;
A = diag(rand(n,1));
B = diag(10*ones(n,1)) + diag(3*ones(n-1,1),1) + diag(2*ones(n-1,1),-1);
A(1,:) = rand(1,n);
A(:,1) = rand(1,n);
[L,U] = my_lu(B);
spy(L*U)

function [L,U] = my_lu(A)
    n = size(A,1);
    L = zeros(size(A));
    U = zeros(size(A));
    A2 = A;
    for k = 1:n
        if A2(k,k) == 0
		    'Encountered 0 pivot. Stopping';
		    return
        end
        L(k,k) =1;
        for i = k+1:n
		    L(i,k) = A2(i,k)/A2(k,k);
        end
        
        for i = k+1:n
		    for j = k+1:n
			    A2(i,j) = A2(i,j) - L(i,k)*A2(k,j);
		    end
        end
    end
U = triu(A2);
end

