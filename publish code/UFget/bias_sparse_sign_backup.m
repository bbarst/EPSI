function Phi = bias_sparse_sign_backup(d,N,zeta,eta)
    rows = randi(d,N*zeta,1); % random coordinates in 1,...,d
    cols = kron((1:N)',ones(zeta,1)); % zeta nonzeros per column
    % signs = (2*randi(2,N*zeta,1) - 3); % uniform random +/-1 values
    signs = rand(N*zeta,1);
    % signs(signs>=0.75) = 1.5;
    % signs(signs<=0.75) = -0.5;
    signs(signs>=0.5) = sqrt(1-eta);
    signs(signs<=0.5) = -sqrt(1-eta);
    Phi = sparse(rows, cols, signs / sqrt(zeta), d, N);
end
