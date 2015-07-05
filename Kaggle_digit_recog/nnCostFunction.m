function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
	

	%calculate h(x)

	 hofx = sigmoid([ones(m,1) , sigmoid([ones(m ,1) X]*(Theta1'))]*Theta2');
	
	 %create labels for every y output 

	 temp_y = zeros(m , num_labels);
	 for iter = 1: num_labels
	 	temp_y(: , iter) = (y==iter);
	 end

	 %calculate cost

	 y_hofx = (-temp_y).*log(hofx);
	 inv_y_hofx = (1-temp_y).*log(1-hofx);

	 tot_sum = y_hofx  - inv_y_hofx;

	 rowsum = sum(tot_sum , 2);
	 J = sum(rowsum)/m;

	 J = J + sum( sum((Theta1.^2))(2:end) )*lambda/(2*m);
	 J = J + sum( sum((Theta2.^2))(2:end) )*lambda/(2*m);

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

		% --------------------------------------- MY CHECK -------------------------------------------------------
			%tmp_thg1 = zeros(size(Theta1));
			%tmp_thg2 = zeros(size(Theta2));
			tmp_a1 = X;
			tmp_a1 = [ones(1,m) ; tmp_a1'];
			
			tmp_z2 = Theta1*tmp_a1;
			tmp_a2 = [ones(1,m);sigmoid(tmp_z2)];
			
			tmp_z3 = Theta2*tmp_a2;
			tmp_a3 = sigmoid(tmp_z3);

			tmp_output = temp_y';

			tmp_del3 = tmp_a3 - tmp_output;
			tmp_del2 = ((Theta2')*tmp_del3).*sigmoidGradient([ones(1,m) ; tmp_z2]) ;

			Theta1_grad = tmp_del2(2:end , : )*(tmp_a1');
			Theta2_grad = tmp_del3*(tmp_a2');


		% ----------------------------------------MY CHECK--------------------------------------------------------	
	

		Theta1_grad = Theta1_grad + lambda*[zeros(size(Theta1,1) , 1) Theta1(: , 2:end)];
		Theta2_grad = Theta2_grad + lambda*[zeros(size(Theta2,1) , 1) Theta2(: , 2:end)];


		Theta2_grad = Theta2_grad/m;
		Theta1_grad = Theta1_grad/m;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

	
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
