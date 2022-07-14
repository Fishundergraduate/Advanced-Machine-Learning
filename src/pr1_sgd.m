n = 200;
d = 4;
x = 3 * (rand(d, n) - 0.5);
y = (2 * x(1, :) - 1 * x(2 , :) + 0.5 + 0.5 * randn(1, n)) > 0;
y = 2 * y -1;
%単純な最急降下法
lr = 10.^(-1 .* 2) ;% learning rate: const

w = ones(d,1); % initial param.
lambda =1;
loop = 200;
nab_J_step = @(w,x,y) exp(-y*(w'*x))/(1+exp(-y*(w'*x))) *(-y).*(x);
J_step = @(w,x,y) log(1+exp(-y*(w'*x)));
J_history = zeros(loop,1);
for t = 1:loop
    %Calc. nabla J
    nab_Jstep = zeros(d,1);
    for i=1:n
        nab_Jstep = nab_Jstep + nab_J_step(w,x(:,i),y(:,i));
    end
    w = w - lr .* (nab_Jstep +2.* lambda.*w) ;
    % Calc. J
    Jstep = 0;
    for i=1:n
        Jstep = Jstep + J_step(w,x(:,i),y(:,i));
    end
    J_history(t) = Jstep + lambda * (w'*w);
end
w;
predict = 2 * (w'* x >0)-1;
fprintf("Accuracy: %.4f \n",sum(predict==y)/n);
J_history = abs(J_history - J_history(loop) + 1e-18);% 1e-18 : To avoid log(0)
semilogy(J_history(1:loop-1));
title('最急降下法での損失関数の推移')
xlabel('時刻')
ylabel('|J(w) - J(w_hat)|')