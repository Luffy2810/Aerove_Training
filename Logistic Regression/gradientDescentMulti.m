function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y);% number of training examples
n=size(X,2)
J_history = zeros(num_iters, 10);

for iter = 1:num_iters
    h = X * theta;
    errors = h - y;
    delta = X' * errors;
    theta = theta - (alpha / m) * delta;

    J_history(iter) = computeCost(X, y, theta);

end
end
