% Machine Learning Online Class
% Exercise 8 | Anomaly Detection and Collaborative Filtering

% Anomaly detection algorithm for server computer behavior

% Data notes:  307 unlabelled examples of server behavior, sepcifically, through-
% put (mb/s) and latency (ms) of response of each server. The vast majority of these 
% examples are assumed to be "normal" (non-anomalous), but a Gaussian model will be
% used to detect any anomalous examples in the dataset. yval contains a validation set.

clear ; close all; clc

% Functions...

function [mu sigma2] = estimateGaussian(X)
	% estimates the parameters of a Gaussian distribution using the data in X
	% Input: Each n-dimensional data point is a row of X. 
	% Output: n-dimensional vector mu, the dataset mean, and variances sigma^2, 
	% an n x 1 vector.
	% Initialize values...
	[m, n] = size(X);
	mu = zeros(n, 1);
	sigma2 = zeros(n, 1);

	% ====================== YOUR CODE HERE ======================
	% Instructions: Compute the mean of the data and the variances
	%               In particular, mu(i) should contain the mean of
	%               the data for the i-th feature and sigma2(i)
	%               should contain variance of the i-th feature.
	%
	mu = 1/m * sum(X);
	sigma2 = 1/m * sum((X - mu).^2);
end

function p = multivariateGaussian(X, mu, Sigma2)
	% computes the probability density function of the multivariate gaussian distribution.
	% Parameters: mu and sigma^2 (or covariance matrix).
	% If Sigma2 is a matrix, it is treated as the covariance matrix. If Sigma2 is a 
	% vector, it is treated as the sigma^2 values of the variances in each dimension (a 
	% diagonal covariance matrix).
	k = length(mu);
	if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
		Sigma2 = diag(Sigma2);
	end
	X = bsxfun(@minus, X, mu(:)');
	p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
		exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));
end

function visualizeFit(X, mu, sigma2)
	% shows you the probability density function of the Gaussian distribution. 
	% Each example has a location (x1, x2) that depends on its feature values.
	[X1,X2] = meshgrid(0:.5:35); 
	Z = multivariateGaussian([X1(:) X2(:)],mu,sigma2);
	Z = reshape(Z,size(X1));
	plot(X(:, 1), X(:, 2),'bx');
	hold on;
	% Do not plot if there are infinities
	if (sum(isinf(Z)) == 0)
		contour(X1, X2, Z, 10.^(-20:3:0)');
	end
	hold off;
end

function [bestEpsilon bestF1] = selectThreshold(yval, pval)
	% finds the best threshold (Epsilon) to use for selecting outliers based on the 
	% results from a validation set (pval) and the ground truth (yval) using
	% an F1 score as the evaluation metric.
	% Initialize values...
	bestEpsilon = 0;
	bestF1 = 0;
	F1 = 0;
	stepsize = (max(pval) - min(pval)) / 1000;
	% Scanning through the ranked pvals in "stepsize" increments...
	for epsilon = min(pval):stepsize:max(pval)
		% predict: binary vector (0's and 1's) of outlier predictions (1 = anomaly)...
		predict = (pval < epsilon);
		pvalPositive = sum(predict == 1);
		yvalPositive = sum(yval == 1);
		truePositive = sum((predict == 1 ) & (yval == 1));
		precision = truePositive / pvalPositive;
		recall = truePositive / yvalPositive;
		% F1 score...
		F1 = 2 * precision * recall / (precision + recall);
		if F1 > bestF1
		   bestF1 = F1;
		   bestEpsilon = epsilon;
		end
	end
end

% Script...

%% ================== Part 1: Load Example Dataset  ===================
%  We start this exercise by using a small dataset that is easy to
%  visualize.
%
%  Our example case consists of 2 network server statistics across
%  several machines: the latency and throughput of each machine.
%  This exercise will help us find possibly faulty (or very fast) machines.
%

fprintf('Visualizing example dataset for outlier detection.\n\n');

%  The following command loads the dataset. You should now have the
%  variables X, Xval, yval in your environment
load('ex8data1.mat');

%  Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause


%% ================== Part 2: Estimate the dataset statistics ===================
%  For this exercise, we assume a Gaussian distribution for the dataset.
%
%  We first estimate the parameters of our assumed Gaussian distribution, 
%  then compute the probabilities for each of the points and then visualize 
%  both the overall distribution and where each of the points falls in 
%  terms of that distribution.
%
fprintf('Visualizing Gaussian fit.\n\n');

%  Estimate my and sigma2
[mu sigma2] = estimateGaussian(X);

%  Returns the density of the multivariate normal at each data point (row) 
%  of X
p = multivariateGaussian(X, mu, sigma2);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Part 3: Find Outliers ===================
%  Now you will find a good epsilon threshold using a cross-validation set
%  probabilities given the estimated Gaussian distribution
% 

pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n\n');

%  Find the outliers in the training set and plot the
outliers = find(p < epsilon);

%  Draw a red circle around those outliers
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Part 4: Multidimensional Outliers ===================
%  We will now use the code from the previous part and apply it to a 
%  harder problem in which more features describe each datapoint and only 
%  some features indicate whether a point is an outlier.
%

%  Loads the second dataset. You should now have the
%  variables X, Xval, yval in your environment
load('ex8data2.mat');

%  Apply the same steps to the larger dataset
[mu sigma2] = estimateGaussian(X);

%  Training set 
p = multivariateGaussian(X, mu, sigma2);

%  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2);

%  Find the best threshold
[epsilon F1] = selectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('# Outliers found: %d\n', sum(p < epsilon));
fprintf('   (you should see a value epsilon of about 1.38e-18)\n\n');
pause



