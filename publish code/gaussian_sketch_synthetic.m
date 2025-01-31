% rng(113);
m = 2000;
n = 50;
s = 300;
q =5;
k=-3;
R = normrnd(0,1,m,n);
[U,~] = qr(R,0);
R = normrnd(0,1,n,n);
[V,~] = qr(R,0);

%change gap
Sigma = diag(logspace(0,k,n));
Sigma(2,2) = Sigma(3,3)+0.0001;


A = U*Sigma*transpose(V);
S = bias_sparse_sign_backup(s,m,8,n/s);
SA = S*A;
SASA = SA'*SA;


U0 = normrnd(0,1,n,q);
A_eig = diag(Sigma);
A_eig = A_eig(1:q);
%PM initialization
for i=1:4
    U0 = A*U0;
    U0 = A'*U0;
    [U0,~] = qr(U0,0);
end


max_iter = 30;
[U1,E1] = power_eig(U0,A,SASA,max_iter,q,diag(Sigma));

[U2,E2] = epsi(U0,A,SASA,max_iter,q,diag(Sigma),1/8,1/4,V);





figure(1)
name = [];
for i=1:q
plot(1:max_iter,log(E2(i,:)))
hold on
name = [name,i];
end
legend('1','2','3','4','5')

figure(2)
name = [];
for i=1:q
plot(1:max_iter,log(E1(i,:)))
name = [name,i];
hold on
end

% plot(1:max_iter,log(E1(1,:)))
% hold on
% plot(1:max_iter,log(E2(1,:)))
% legend('p','d')






function [u1,E] = debias_eig2(u0,A,SASA,max_iter,q,A_eig)
n = size(A,2);
E = zeros(q,max_iter);
[U,L] = eig(SASA);
for i=1:max_iter
    LL = inv(u0'*u0)*(A*u0)'*(A*u0);
    P = (eye(n)-u0*inv(u0'*u0)*u0');
    du = sylvester(P*SASA,LL,A'*(A*u0)-u0*LL);
    [u1,~] = qr(u0+du,0);
    u0=u1;
    M = vecnorm(A*u0)';
    M = sort(M,'descend');
    E(:,i) =   abs(M-A_eig(1:q));
end

end


function [u1,E] = epsi(u0,A,SASA,max_iter,q,A_eig,c1,c2,realV)
n = size(A,2);
E = zeros(q,max_iter);
[U,L] = eig(SASA);
AA = A'*A;
U1 = realV(:,1:q);
U2 = realV(:,q+1:end);
V0 = ones(1,q);
V_last1 = ones(1,q);
V_last2 = ones(1,q);
U4 = [U1(:,1),U1(:,4:5)];
U5 = U1(:,2:3);
for i=1:max_iter
    U = zeros(n,1);
    % EPSI
    for j=1:q
        u = u0(:,j);
        ll = (A*u)'*(A*u);
        % F_g = ((eye(n)-2*u*u')*(AA-ll*eye(n))+c1*u*u'+c2*u*((u'*U)*U')+c2/2*(U'*u)'*(U'*u));
        % du = inv(F_g)*(A'*(A*u)-ll*u+c2/2*(U'*u)'*(U'*u)*u);
        F_g = (ll*eye(n)-(eye(n)-U*U')*SASA*(eye(n)-U*U'));
        du = inv(F_g)*(ll*u-A'*(A*u));

        
        u1(:,j) = u - du;
        % make u1 orthogonal to former U
        % u1(:,j) = (eye(n)-U*U')*u1(:,j);
        u1(:,j) = u1(:,j)/norm(u1(:,j));

        % u1 = inv(norm(u)*eye(n)-SASA)*(A'*(A*u0)-SASA*u0);
        U(:,j) = u1(:,j);
        U(:,j) = (eye(n)-U*U')*U(:,j);
        U(:,j) = U(:,j)/norm(U(:,j));
        %fprintf('\n debiased error: %.4f\n', norm(l0-1));
        
    end

    [U,~] = qr(u1,0);
    % UUU = vecnorm(U2'*U)
    
    %eigendecomposition
    [t,l] = eig((A*U)'*(A*U));
    [t,l] = sort_by_val(t,l);
    
    
    u0=u1;
    u0=U*t;


    M = vecnorm(A*u0)';
    M = sort(M,'descend');
    E(:,i) =   abs(M-A_eig(1:q));
end
end



function [U,E] = debias_eig_n(u0,A,SASA,max_iter,q,A_eig,c1,c2)
n = size(A,2);
E = zeros(q,max_iter);
[U,L] = eig(SASA);
AA = A'*A;
for i=1:max_iter
    U = zeros(n,1);
    for j=1:q
        u = u0(:,j);
        ll = (A*u)'*(A*u);
        F_g = ((eye(n)-2*u*u')*(SASA-ll*eye(n))+c1*u*u'+c2*u*((u'*U)*U')+c2/2*(U'*u)'*(U'*u));
        % F_g = (SASA-ll*eye(n)+c2*u*((u'*U)*U')+c2/2*(U'*u)'*(U'*u));
        % F_g = (SASA-ll*eye(n)+c2/2*U*U');
        u1(:,j) = u - inv(F_g)*(A'*(A*u)-ll*u+c2/2*(U'*u)'*(U'*u)*u);
        % u1 = inv(norm(u)*eye(n)-SASA)*(A'*(A*u0)-SASA*u0);
        u1(:,j) = u1(:,j)/norm(u1(:,j));
        U(:,j) = u1(:,j);
        %fprintf('\n debiased error: %.4f\n', norm(l0-1));
        
    end
    [u1,~] = qr(u1,0);
    
    [u0,~] = eig(u1'*A'*A*u1);
    u0 = u1*u0;
    M = vecnorm(A*u0)';
    M = sort(M,'descend');
    E(:,i) =   abs(M-A_eig(1:q));
end
end

function [u1,E] = lazy_debias(u0,A,SASA,max_iter,q,A_eig,c1,c2)
n = size(A,2);
E = zeros(q,max_iter);
[U,L] = eig(SASA);
V = zeros(n,1);
M = SASA;
hat_A = A'*A;
for i=1:q
    u = u0(:,i);
    for j = 1:max_iter
        ll = norm(u'*A'*(A*u));
        F_g = M-ll*eye(n);
        u1 = u-inv(F_g)*(hat_A*u-ll*u);
        u1 = u1/norm(u1);
        u = u1;
        E(i,j) = abs(A_eig(i)-norm(A*u));
    end
    u = (eye(n)-V*V')*u;
    u = u/norm(u);
    V(:,i) = u;
    M =(eye(n)-u*u')*M*(eye(n)-u*u');
    hat_A = (eye(n)-u*u')*hat_A*(eye(n)-u*u');
end
end


function [u1,E] = power_eig(u0,A,SASA,max_iter,q,A_eig)
n = size(A,2);
E = zeros(q,max_iter);
for i=1:max_iter
    
    u1 = A'*(A*u0);
    [u1,~] = qr(u1,0);
    u0 = u1;
    %fprintf('\n power method error: %.4f\n', norm(norm(A*uk)-1));
    E(:,i) =   abs(vecnorm(A*u1)'-A_eig(1:q));
end

end

function [u1,l1,E] = qr_eig(u0,A,SASA,max_iter,q)
n = size(A,2);
E = zeros(q,max_iter);
AA = A'*A;
Q0 = eye(n);
for i=1:max_iter
    [Q,R] = qr(AA-AA(2,2)*eye(n));
    Q0 = Q0*Q;
    AA = R*Q+AA(2,2)*eye(n);
    E(i) = norm(AA(1,1)-1);
end
u1 = Q0(:,1);
l1 = norm(A*u1);
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