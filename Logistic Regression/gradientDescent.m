function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    h = X * theta;
    Error = (h - y);
    Errors = (h - y).*X(:,2);
    t1 = theta(1,1) - alpha*sum(Error)/m;
    t2 = theta(2,1) - alpha*sum(Errors)/m;
    theta(1,1)= t1;
    theta(2,1) = t2;
    J_history(iter,1) = computeCost(X, y, theta);



end
J_history(iter,1) = computeCost(X, y, theta)

end
