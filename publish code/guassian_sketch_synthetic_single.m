m = 2000;
n = 50;
s = 300;

k=-3;
R = normrnd(0,1,m,n);
[U,~] = qr(R,0);
R = normrnd(0,1,n,n);
[V,~] = qr(R,0);

Sigma = diag(logspace(0,0+k,n));



%adjust first eigen gap
% Sigma(2,2) = Sigma(1,1)*0.95;

eig_A = diag(Sigma);
A = U*Sigma*transpose(V);

%first sketch
S = bias_sparse_sign_backup(s,m,8,n/s);
SA = S*A;
SASA = SA'*SA;

u0 = normrnd(0,1,n,1);
for i=1:5
    u0 = SASA*u0;
    u0 = u0/norm(u0);
    l0 = norm(SASA*u0);
end

max_iter = 100;
[~,~,E1] = newton(u0,A,SASA,max_iter,eig_A,V);
[U2,~,E2] = debias_eig(u0,A,S,SASA,max_iter,eig_A,V);
[~,~,E3] = debias_eig2(u0,A,SASA,max_iter,eig_A,1/8,V);




% plot(1:max_iter,log(E1(1,:)))
% hold on
% plot(1:max_iter,log(E2(1,:)))
% hold on
% plot(1:max_iter,log(E3(1,:)))
% legend('newton','debias','debias 2')

%plot E2
for i=1:5

    plot(1:max_iter,log(E2(i,:)))
    hold on
end
legend('1','2','3','4','5','6','7','8','9','10')



function [u1,l1,E] = debias_eig(u0,A,S,SASA,max_iter,eig_A,vec_A)
n = size(A,2);
m = size(A,1);
E = zeros(1,max_iter);
[U,L] = eig(SASA);
E = zeros(10,max_iter);
for i=1:max_iter
    
    

    ll = norm(A*u0)^2/norm(u0)^2;
    t = chol(A'*A);
    % u1 = inv(norm(u)*eye(n)-SASA)*(A'*(A*u0)-SASA*u0);
    % u1 = u0 - inv(SASA-ll*eye(n)-0.4*u0*u0')*(A'*(A*u0)-ll*u0+0.2*(u0'*u0-1)*u0);
    % u1 = inv(ll*eye(n)-(SASA))*(A'*(A*u0)-(SASA)*u0);
    u1 = u0 - inv(ll*eye(n)-(SASA))*(ll*u0-A'*(A*u0));

    u1 = u1/norm(u1);
    
    l1 = norm(A*u1)/norm(u1);
    l0=l1;
    u0=u1;
    %fprintf('\n debiased error: %.4f\n', norm(l0-1));
    % E(i) =   norm(l0-eig_A(1));
    for j =1:10
        E(j,i) = norm(vec_A(:,j:end)'*u1);
    end
end

end



function [u1,l1,E] = debias_eig2(u0,A,SASA,max_iter,eig_A,c,vec_A)
n = size(A,2);
E = zeros(5,max_iter);
for i=1:max_iter
    ll = norm(u0'*(A'*(A*u0)))/norm(u0'*u0);
    
    F_g  =(eye(n)-2*u0*u0')*(SASA-ll*eye(n))+c*u0*u0';
    temp = (A'*(A*u0)-ll*u0+c/2*(u0'*u0-1)*u0);
    u1 = u0-inv(F_g)*temp;
    u1 = u1/norm(u1);
    u0 = u1;
    for j =1:5
        E(j,i) = norm(vec_A(:,j:end)'*u1);
    end
end
l1 = norm(A*u1);

end

function [u1,l1,E] = power_eig(u0,A,SASA,max_iter,eig_A,vec_A)
n = size(A,2);
E = zeros(5,max_iter);
invA = inv(A'*A);
for i=1:max_iter
    
    u1 = A'*(A*u0);
    u1 = u1/norm(u1);
    u0 = u1;
    %fprintf('\n power method error: %.4f\n', norm(norm(A*uk)-1));
    for j =1:5
        E(j,i) = norm(vec_A(:,j)'*u1);
    end
end
l1 = norm(A*u1);

end


function [u1,l1,E] = newton(u0,A,c,max_iter,eig_A,vec_A)
n = size(A,2);
E = zeros(5,max_iter);
AA = A'*A;
for i=1:max_iter
    ll = norm(u0'*(A'*(A*u0)))/norm(u0'*u0);
    F_g  =(eye(n)-2*u0*u0')*(AA-ll*eye(n))+c*u0*u0';
    u1 = u0-inv(F_g)*(A'*(A*u0)-ll*u0+c/2*(u0'*u0-1)*u0);
    % u1 = u1/norm(u1);
    u0 = u1;
    % E(i) =   norm(norm(A*u1)-eig_A(1));
    for j =1:5
        E(j,i) = norm(vec_A(:,5*j)'*u1);
    end
end
l1 = norm(A*u1);

end

