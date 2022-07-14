n = 200;% data amount
d = 4;% data dimension
c = 3;% class
x = 3 * (rand(n, d) - 0.5);
W = [2, -1, 0.5;
-3, 2, 1;
1, 2, 3];
[maxlogit, y] =max( [x(:, 1:2), ones(n, 1)] * W' + 0.5 * randn(n, c), [], 2);
y_onehot = (y == 1:c)*2-1;

% multiclass SGD
lr = 10.^(-1 .* 2) ;% learning rate: const
x = x';
y = y';
y_onehot = y_onehot';
w = ones(d,c); % initial param.
lambda =1;
loop = 200;
nab_J_step = @(w,x,y) exp(-y'*(w'*x))/(1+exp(-y'*(w'*x))) *(-y).*(x);
J_step = @(w,x,y) log(1+exp(-y*(w'*x)));
J_history = zeros(loop,c);
for t = 1:loop
    for class = 1:c
        nab_Jstep = zeros(d,1);
        for i=1:n
            nab_Jstep = nab_Jstep + nab_J_step(w(:,class),x(:,i),y_onehot(class,i));
        end
        w(:,class) = w(:,class) - lr .* (nab_Jstep +2.* lambda.*w(:,class)) ;
        Jstep = 0;
        for i=1:n
            Jstep = Jstep + J_step(w(:,class),x(:,i),y_onehot(class,i));
        end
        J_history(t,class) = Jstep + lambda * (w(:,class)'*w(:,class));
    end
end
w
predict = w'*x;
for t = 1:loop
    [M, I] = max(predict(:,t));
    label = -1 .* ones(1,c);
    label(1,I) = 1;
    predict(:,t) = label(1,:);
end
result = (predict == y_onehot)*1;
fprintf("Accuracy: %.4f \n",sum(result(:))/(c*n));
J_history = abs(J_history - J_history(loop,:) + 1e-18.*ones(size(J_history)));% 1e-18 : To avoid log(0)
semilogy(J_history(1:loop-1,1));
hold on
semilogy(J_history(1:loop-1,2));
semilogy(J_history(1:loop-1,3));
legend('class1', 'class2', 'class3');
title('最急降下法での損失関数の推移')
xlabel('時刻')
ylabel('|J(w) - J(w_hat)|')
hold off