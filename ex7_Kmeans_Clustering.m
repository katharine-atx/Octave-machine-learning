% Machine Learning Online Class
%  Exercise 7 | Principle Component Analysis and K-Means Clustering

% Data notes: K-means clustering is performed on a set of 300 training
% examples in 2D matrix X.

clear ; close all; clc

% Functions...

function idx = findClosestCentroids(X, centroids)
	% returns the closest centroids in idx for a dataset X where each row is a 
	% single example. idx = m x 1 vector of centroid assignments (i.e. each 
	% entry in range [1..K])
	% Set K
	K = size(centroids, 1);
	% Initialize values...
	idx = zeros(size(X,1), 1);
	m = size(X,1); % number of training examples
	% Calculate cost across m training examples for centroid one...
	for i = 1:m
		k = 1;
		cost1 = (X(i,:) - centroids(k,:)) * (X(i,:) - centroids(k,:))';
		idx(i) = k;
		% And for the remaining 2:K centroids, storing the minimum k index...
		for k = 2:K
			cost2 = (X(i,:) - centroids(k,:)) * (X(i,:) - centroids(k,:))';
			if cost2 < cost1
				cost1 = cost2;
				idx(i) = k;
			end
		end
	end
end       

function centroids = kMeansInitCentroids(X, K)
	% returns K initial centroids to be used with the K-Means on the dataset X
	% Initialize values...
	centroids = zeros(K, size(X, 2));
	m = size(X, 1);  %number of training examples
	% Pick K random indices from m...
	randomIndex = randperm(m, K);
	% Set centroids equal to those training examples (rows of X)...
	centroids = X(randomIndex, :);
end

function centroids = computeCentroids(X, idx, K)
	% returns the new centroids by computing the means of the data points 
	% assigned to each centroid. It is given a dataset X where each row is a 
	% single data point, a vector idx of centroid assignments (i.e. each entry 
	% in range [1..K]) for each example, and K, the number of centroids.
	% Returns a matrix centroids, where each row of centroids is the mean of 
	% the data points assigned to it.
	% Initialize values...
	[m n] = size(X);
	centroids = zeros(K, n);
	% Calculate the mean for a particular centroid (match by index)...
	for c = 1:K
		match = idx == c;
		centroids(c, :) = sum(X(match, :)) / sum(match);
	end
end

function plotProgresskMeans(X, centroids, previous, idx, K, i)
	% plots the data points with colors assigned to each centroid. With the previous
	% centroids, it also plots a line between the previous locations and current 
	% locations of the centroids. Note: intended for 2D data.
	% Plot the examples
	plotDataPoints(X, idx, K);
	% Plot the centroids as black x's
	plot(centroids(:,1), centroids(:,2), 'x', ...
		 'MarkerEdgeColor','k', ...
		 'MarkerSize', 10, 'LineWidth', 3);
	% Plot the history of the centroids with lines
	for j=1:size(centroids,1)
		drawLine(centroids(j, :), previous(j, :));
	end
	% Title
	title(sprintf('Iteration number %d', i))
end

function [centroids, idx] = runkMeans(X, initial_centroids, ...
                                      max_iters, plot_progress)
	% runs the K-Means algorithm on data matrix X, where each row of X is a 
	% single example. It uses initial_centroids used as the initial centroids. 
	% max_iters specifies the total number of iterations of K-Means to execute.
	% plot_progress is a true/false flag that indicates if the function should 
	% also plot its progress as the learning happens. Returns centroids, a Kxn 
	% matrix of the computed centroids and idx, a m x 1 vector of centroid 
	% assignments (i.e. each entry in range [1..K])
	% Set default value for plot progress...
	if ~exist('plot_progress', 'var') || isempty(plot_progress)
		plot_progress = false; %Switch to true or false, depending on preference.
	end
	% Plot the data if we are plotting progress
	if plot_progress
		figure;
		hold on;
	end
	% Initialize values
	[m n] = size(X);
	K = size(initial_centroids, 1);
	centroids = initial_centroids;
	previous_centroids = centroids;
	idx = zeros(m, 1);
	% Run K-Means
	for i=1:max_iters
		% Output progress
		fprintf('K-Means iteration %d/%d...\n', i, max_iters);
		if exist('OCTAVE_VERSION')
			fflush(stdout);
		end
		% For each example in X, assign it to the closest centroid
		idx = findClosestCentroids(X, centroids);
		% Optionally, plot progress here
		if plot_progress
			plotProgresskMeans(X, centroids, previous_centroids, idx, K, i);
			previous_centroids = centroids;
			fprintf('Press enter to continue.\n');
			pause;
		end
		% Given the memberships, compute new centroids
		centroids = computeCentroids(X, idx, K);
	end
	% Hold off if we are plotting progress
	if plot_progress
		hold off;
	end
end

% Script...

%% ================= Part 1: Find Closest Centroids ====================
%  To help you implement K-Means, we have divided the learning algorithm 
%  into two functions -- findClosestCentroids and computeCentroids. In this
%  part, you shoudl complete the code in the findClosestCentroids function. 
%
fprintf('Finding closest centroids.\n\n');

% Load an example dataset that we will be using
load('ex7data2.mat');

% Select an initial set of centroids
K = 3; % 3 Centroids
initial_centroids = [3 3; 6 2; 8 5];

% Find the closest centroids for the examples using the
% initial_centroids
idx = findClosestCentroids(X, initial_centroids);

fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d', idx(1:3));
fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ===================== Part 2: Compute Means =========================
%  After implementing the closest centroids function, you should now
%  complete the computeCentroids function.
%
fprintf('\nComputing centroids means.\n\n');

%  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K);

fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\n(the centroids should be\n');
fprintf('   [ 2.428301 3.157924 ]\n');
fprintf('   [ 5.813503 2.633656 ]\n');
fprintf('   [ 7.119387 3.616684 ]\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 3: K-Means Clustering ======================
%  After you have completed the two functions computeCentroids and
%  findClosestCentroids, you have all the necessary pieces to run the
%  kMeans algorithm. In this part, you will run the K-Means algorithm on
%  the example dataset we have provided. 
%
fprintf('\nRunning K-Means clustering on example dataset.\n\n');

% Load an example dataset
load('ex7data2.mat');

% Settings for running K-Means
K = 3;
max_iters = 10;

% For consistency, here we set centroids to specific values
% but in practice you want to generate them automatically, such as by
% settings them to be random examples (as can be seen in
% kMeansInitCentroids).
initial_centroids = [3 3; 6 2; 8 5];

% Run K-Means algorithm. The 'true' at the end tells our function to plot
% the progress of K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: K-Means Clustering on Pixels ===============
%  In this exercise, you will use K-Means to compress an image. To do this,
%  you will first run K-Means on the colors of the pixels in the image and
%  then you will map each pixel on to it's closest centroid.
%  
%  You should now complete the code in kMeansInitCentroids.m
%

fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');

%  Load an image of a bird
A = double(imread('bird_small.png'));

% If imread does not work for you, you can try instead
%   load ('bird_small.mat');

A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

% Size of the image
img_size = size(A);

% Reshape the image into an Nx3 matrix where N = number of pixels.
% Each row will contain the Red, Green and Blue pixel values
% This gives us our dataset matrix X that we will use K-Means on.
X = reshape(A, img_size(1) * img_size(2), 3);

% Run your K-Means algorithm on this data
% You should try different values of K and max_iters here
K = 16; 
max_iters = 10;

% When using K-Means, it is important the initialize the centroids
% randomly. 
% You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 5: Image Compression ======================
%  In this part of the exercise, you will use the clusters of K-Means to
%  compress an image. To do this, we first find the closest clusters for
%  each example. After that, we 

fprintf('\nApplying K-Means to compress an image.\n\n');

% Find closest cluster members
idx = findClosestCentroids(X, centroids);

% Essentially, now we have represented the image X as in terms of the
% indices in idx. 

% We can now recover the image from the indices (idx) by mapping each pixel
% (specified by it's index in idx) to the centroid value
X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));


fprintf('Program paused. Press enter to continue.\n');
pause;

