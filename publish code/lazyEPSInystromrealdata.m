clc;
clear;

% data = 'amazon0302';
data='web-Stanford';
% data = 'email-Enron';
S = UFget(['SNAP/' data]);


A = S.A;
q=10;
[U,Sigma,V]=svds(A,10);




%% Random SVD Initialization
[~,~,U0] = random_svd(A,q,10);

max_iter = 50;
[U2,E2] = LazyEPSI(U0,A,max_iter,q,diag(Sigma),1/8,1/4,V);
[U1,E1] = RandSubspace(U0,A,max_iter,q,diag(Sigma),1/8,1/4,V);





colors = parula(q); 

%%%%%%%%%%%%%%%%%%%%%
% EPSI VS Subspace Iteration
%%%%%%%%%%%%%%%%%%%%%
figure('Position', [0, 0, 2400, 800])

gca=subplot(121);
box on
name = [];
for i=1:q
semilogy(1:max_iter,(E2(i,:)),'*-','Color', colors(i,:),'MarkerSize',5,'linewidth',2)
name = [name,string(i)+"-th"];
hold on
end
gca.LineWidth = 4;
gca.FontSize=20;
gca.XTick = [0:10:50];
lgd = legend(name,'FontSize',16);
lgd.LineWidth = 2;
title('Lazy-EPSI','FontSize',60)
%xlabel('Iteration','FontSize',36)
%hy = ylabel('Error','FontSize',36);
%hy.Position = [-3.0    0  -1.0000];

ylim([1e-10 1]);

grid on;


gca=subplot(122);
box on
name = [];
for i=1:q
semilogy(1:max_iter,(E1(i,:)),'*-','Color', colors(i,:),'MarkerSize',5,'linewidth',2)
name = [name,string(i)+"-th"];
hold on
end
gca.XTick = [0:10:50];
gca.LineWidth = 4;
gca.FontSize=20;
lgd = legend(name,'FontSize',16);
lgd.LineWidth = 2;
title('Subspace Iteration','FontSize',60)
%xlabel('Iteration','FontSize',36)
%ylabel('Error','FontSize',36)
ylim([1e-10 1]);

grid on;

saveas(gcf,['lazyepsi' data  '.png'])




function [u1,E] = RandSubspace(u0,A,max_iter,q,A_eig,c1,c2,real_V)
    [m, n] = size(A);
    E = zeros(q,max_iter);
    %Omega = randn(n, q + 1);   
    %Y = A * Omega;
    Q = u0;
    for i = 1:max_iter
        Y = A*(A'*Q);
        [Q, ~] = qr(Y, 0);
        B = Q' * A;
        [U_tilde, S, V] = svd(B, 'econ');
        %U = Q * U_tilde;
        u1 = V(:,1:q);
        M = vecnorm(A*u1)';
        M = sort(M,'descend');
        E(:,i) =   abs(M-A_eig(1:q));
        % E(:,i) = vecnorm(A'*(A*u1)-u1*diag(A_eig(1:q).^2))';
        % E(:,i) = ((ones(1,q)-vecnorm(real_V'*u1).^2).^0.5)';
    end

end



function [U, Lambda_hat] = randomizedNystrom(A, ell, q)
% RANDOMIZEDNYSTROM  Implements a randomized Nyström approximation 
% for a positive semidefinite matrix ATA.
%
% Input:
%   A   : n-by-n  matrix (or, more generally, a Hermitian PSD)
%   ell : approximation rank (<= n)
%   k   : target rank
%
% Output: nystrom approximation for ATA
%   U           : n-by-ell orthonormal matrix
%   Lambda_hat  : ell-by-ell diagonal matrix containing approximate eigenvalues

    % 1) Sample a random Gaussian test matrix and perform thin QR
    k=2;
    j = 0;
    [~, n] = size(A);
    Omega = randn(n, ell);             % n-by-ell
  
    % 2) Form Y = A * Omega
    Y =Omega;                     % n-by-ell
    for i = 1:k
        [Omega,~] = qr(Y,'econ');
        Y = A'*(A * Omega);
        for a_a = 1:j
            Y = A'*(A * Y);
        end
    end
    
    % 3) Stability shift: Y_nu = Y + nu * Omega, where nu = eps(norm(Y,'fro'))
    nu = eps(norm(Y, 'fro'));
    Y_nu = Y + nu * Omega;            % n-by-ell
    
    % 4) Compute C = chol(Omega' * Y_nu) and B = Y_nu / C
    C = chol(Omega' * Y_nu);          % C is ell-by-ell
    B = Y_nu / C;                     % B is n-by-ell
    
    % 5) Compute thin SVD of B
    [U, Sigma, ~] = svd(B, 'econ');   % U is n-by-ell, Sigma is ell-by-ell
    
    % 6) Shift the singular values and retain nonnegative part:
    %    Lambda_hat = max(0, Sigma^2 - nu*I).
    %    We store this as a diagonal matrix for convenience.
    sigma2 = (diag(Sigma) - nu).^(2/(1+j));   % 1-by-ell
    sigma2_shifted = max(sigma2, 0); % clamp below at 0
    Lambda_hat = diag(sigma2_shifted)*(1-2*q/(ell-q));
    
    % That’s it! A_nys = U * Lambda_hat * U' is the approximate rank-ell 
    % Nyström factorization for the PSD matrix A.

end


function [u1,E] = LazyEPSI(u0,A,max_iter,q,A_eig,c1,c2,real_V)
% c1 = 1e-10;
c1=0;
n = size(A,2);
E = zeros(q,max_iter);


[Unys,Lambda_hat]= randomizedNystrom(A, 200,q);
UnysT = Unys';
Lambda_hat = Lambda_hat.^(1);

for i=1:max_iter
    U = zeros(n,1);
    % EPSI
    for j=1:q
        u = u0(:,j);
        ll = u'*(A'*(A*u));
        % F_g = ((eye(n)-2*u*u')*(AA-ll*eye(n))+c1*u*u'+c2*u*((u'*U)*U')+c2/2*(U'*u)'*(U'*u));
        % du = inv(F_g)*(A'*(A*u)-ll*u+c2/2*(U'*u)'*(U'*u)*u);
        %F_g = ((eye(n)-U*U')*ASSA*(eye(n)-U*U')-ll*eye(n));
        %du = inv(F_g)*(A'*(A*u)-ll*u);
        
        
        
        
        
        sketch_error = ((A'*(A*u))-ll*u); 

        %% -I + W(-sigma *Lambda ^-1 +W^TW)^{-1}W^T
        % W=(I-UU^T)U_nys
        WTsketch_error = UnysT*sketch_error-UnysT*(U*(U'*sketch_error));
        % compute (-sigma *Lambda ^-1 +W^TW)^{-1}
        Utmp = U'*Unys;
        WTW=  UnysT*Unys-Utmp'*Utmp;
        F_g = WTW-diag((1./diag(Lambda_hat-c1))*(ll-c1));
        % du1 = Unys*((WTsketch_error\F_g)');
        % du1 = du1-U*(U'*du1);
        % du = du1-u;

        
        tmp = Unys*(F_g\WTsketch_error);

        du1 = sketch_error-(tmp-U*(U'*tmp));
        du1 = -du1/ll;


        u1(:,j) =  u-du1;
        % make u1 orthogonal to former U
        u1(:,j) = u1(:,j)-U*(U'*u1(:,j));
        u1(:,j) = u1(:,j)/norm(u1(:,j));
        % u1 = inv(norm(u)*eye(n)-SASA)*(A'*(A*u0)-SASA*u0);
        U(:,j) = u1(:,j);
        %fprintf('\n debiased error: %.4f\n', norm(l0-1));
        
    end
    [U,~] = qr(u1,0);
    %eigendecomposition
    [t,l] = eig((A*U)'*(A*U));
    [t,l] = sort_by_val(t,l);
    u0=u1;
    u0=U*t;
 
  

    M = vecnorm(A*u0)';
    M = sort(M,'descend');
    E(:,i) =   abs(M-A_eig(1:q));
    % E(:,i) = vecnorm(A'*(A*u0)-u0*diag(A_eig(1:q).^2))';
    % E(:,i) = ((ones(1,q)-vecnorm(real_V'*u1).^2).^0.5)';
end
end




function [U, S, V] = random_svd(A, k, p)
    % A: 输入矩阵 (m x n)
    % k: 目标低维子空间维度
    % p: 过采样参数 (推荐值: p >= 5)

    % Step 1: 随机投影
    [m, n] = size(A);
    Omega = randn(n, k + p);   % 生成随机矩阵
    Y = A * Omega;            % 投影到随机子空间
    %for i = 1:5
    %    Y = A*(A'*Y);
    %end

    % Step 2: 正交化 (QR分解)
    [Q, ~] = qr(Y, 0);        % 计算正交基矩阵 Q

    % Step 3: 将 A 投影到低维空间
    B = Q' * A;               % B 是低维近似矩阵 (k+p x n)
    

    % Step 4: 对较小矩阵 B 计算精确 SVD
    [U_tilde, S, V] = svd(B, 'econ');

    % Step 5: 恢复 U 到原空间
    U = Q * U_tilde;
    V=V(:,1:k);

    % 返回结果
end


function [Vs, Ds] = sort_by_val(V, D)
    %%% simply sort the [eigenvectors, eigenvalues] by the module of
    %%% eigenvalues
    eigenvalues = abs(diag(D));
    [~, indices] = sort(eigenvalues, 'descend');
    Vs = V(:, indices);
    D = diag(D);
    Ds = diag(D(indices));
end