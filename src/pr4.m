clear;
n = 200;
x = 3 * (rand(n, 4) - 0.5);
y = (2 * x(:, 1) - 1 * x(:,2) + 0.5 + 0.5 * randn(n, 1)) > 0;
y = 2 * y -1;
lambda = 1;
%cvx model
cvx_begin quiet
    variable w(4); 
    variable xi(n); 
    variable e(4);
    dual variables p q r s;
    minimize (sum(xi) + lambda*sum(e)); 
    subject to
       p: xi >= max(0, 1 - y .* (x * w));
       r: e >= abs(w);
cvx_end
w
xi
e