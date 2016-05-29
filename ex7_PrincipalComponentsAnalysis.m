%  Machine Learning Online Class
%  Exercise 7 | Principle Component Analysis and K-Means Clustering
%
% Data notes: this script compresses unlabelled data in X via 
% principal components analysis, then implements K-Means
% clustering. Also included are functions for mean normalization,
% approximation (recovery) of compressed data and visualization 
% of K-means progression.

clear ; close all; clc

% Functions...

function [h, display_array] = displayData(X, example_width)
	% displays 2D data stored in X in a grid. 
	% Set example_width automatically if not passed in
	if ~exist('example_width', 'var') || isempty(example_width) 
		example_width = round(sqrt(size(X, 2)));
	end
	% Gray Image
	colormap(gray);
	% Compute rows, cols
	[m n] = size(X);
	example_height = (n / example_width);
	% Compute number of items to display
	display_rows = floor(sqrt(m));
	display_cols = ceil(m / display_rows);
	pad = 1; % Between images padding
	% Setup blank display
	display_array = - ones(pad + display_rows * (example_height + pad), ...
						   pad + display_cols * (example_width + pad));
	% Copy each example into a patch on the display array
	curr_ex = 1;
	for j = 1:display_rows
		for i = 1:display_cols
			if curr_ex > m, 
				break; 
			end
			% Copy the patch
			
			% Get the max value of the patch
			max_val = max(abs(X(curr_ex, :)));
			display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
						  pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
							reshape(X(curr_ex, :), example_height, example_width) / max_val;
			curr_ex = curr_ex + 1;
		end
		if curr_ex > m, 
			break; 
		end
	end
	% Display Image
	h = imagesc(display_array, [-1 1]);
	% Do not show axis
	axis image off
	drawnow;
end

function drawLine(p1, p2, varargin)
	% Draws a line from point p1 to point p2 and holds the current figure
	plot([p1(1) p2(1)], [p1(2) p2(2)], varargin{:});
end

function plotDataPoints(X, idx, K)
	% plots data points in X, coloring them so that those with the same index assignments 
	% in idx have the same color.
	% Create palette...
	palette = hsv(K + 1);
	colors = palette(idx, :);
	% Plot the data
	scatter(X(:,1), X(:,2), 15, colors);
end

function [X_norm, mu, sigma] = featureNormalize(X)
	% returns a normalized version of X where the mean value of each feature is 0 and the 
	% standard deviation is 1. 
	mu = mean(X);
	X_norm = bsxfun(@minus, X, mu);
	sigma = std(X_norm);
	X_norm = bsxfun(@rdivide, X_norm, sigma);
end

function [U, S] = pca(X)
	% Principal Components Analysis: computes eigenvectors of the covariance 
	% matrix of X. Returns the eigenvectors U, the eigenvalues (on diagonal) in S
	% Initiate values
	[m, n] = size(X);
	U = zeros(n);
	S = zeros(n);
	% Calculate covariance matrix and use singular value decomposition function, svd()...
	covMatrix = 1/m * X' * X;
	[U, S, V] = svd(covMatrix);
end

function Z = projectData(X, U, K)
	% computes the projection of the normalized inputs X into the reduced dimensional space 
	% spanned by the first K columns of U. It returns the projected examples in Z.
	% Initialize values...
	Z = zeros(size(X, 1), K);
	% Calculate Z...
	Ureduce = U(:, 1:K);
	Z = (Ureduce' * X')';
end

function X_rec = recoverData(Z, U, K)
	% recovers an approximation the original data that has been reduced to K dimensions. 
	% It returns the approximate reconstruction in X_rec.
	% Initialize values....
	X_rec = zeros(size(Z, 1), size(U, 1));
	% Calculate Ureduce and X_rec...
	Ureduce = U(:, 1:K);
	% Reconstructing X, transposing to match original X dimensions...
	X_rec = (Ureduce * Z')';
end

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

%% ================== Part 1: Load Example Dataset  ===================
%  We start this exercise by using a small dataset that is easily to
%  visualize
%
fprintf('Visualizing example dataset for PCA.\n\n');

%  The following command loads the dataset. You should now have the 
%  variable X in your environment
load ('ex7data1.mat');

%  Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bo');
axis([0.5 6.5 2 8]); axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =============== Part 2: Principal Component Analysis ===============
%  You should now implement PCA, a dimension reduction technique. You
%  should complete the code in pca.m
%
fprintf('\nRunning PCA on example dataset.\n\n');

%  Before running PCA, it is important to first normalize X
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);

%  Compute mu, the mean of the each feature

%  Draw the eigenvectors centered at mean of data. These lines show the
%  directions of maximum variations in the dataset.
hold on;
drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;

fprintf('Top eigenvector: \n');
fprintf(' U(:,1) = %f %f \n', U(1,1), U(2,1));
fprintf('\n(you should expect to see -0.707107 -0.707107)\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 3: Dimension Reduction ===================
%  You should now implement the projection step to map the data onto the 
%  first k eigenvectors. The code will then plot the data in this reduced 
%  dimensional space.  This will show you what the data looks like when 
%  using only the corresponding eigenvectors to reconstruct it.
%
%  You should complete the code in projectData.m
%
fprintf('\nDimension reduction on example dataset.\n\n');

%  Plot the normalized dataset (returned from pca)
plot(X_norm(:, 1), X_norm(:, 2), 'bo');
axis([-4 3 -4 3]); axis square

%  Project the data onto K = 1 dimension
K = 1;
Z = projectData(X_norm, U, K);
fprintf('Projection of the first example: %f\n', Z(1));
fprintf('\n(this value should be about 1.481274)\n\n');

X_rec  = recoverData(Z, U, K);
fprintf('Approximation of the first example: %f %f\n', X_rec(1, 1), X_rec(1, 2));
fprintf('\n(this value should be about  -1.047419 -1.047419)\n\n');

%  Draw lines connecting the projected points to the original points
hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');
for i = 1:size(X_norm, 1)
    drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
end
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 4: Loading and Visualizing Face Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment
%
fprintf('\nLoading face dataset.\n\n');

%  Load Face dataset
load ('ex7faces.mat')

%  Display the first 100 faces in the dataset
displayData(X(1:100, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
%  Run PCA and visualize the eigenvectors which are in this case eigenfaces
%  We display the first 36 eigenfaces.
%
fprintf(['\nRunning PCA on face dataset.\n' ...
         '(this mght take a minute or two ...)\n\n']);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X);

%  Run PCA
[U, S] = pca(X_norm);

%  Visualize the top 36 eigenvectors found
displayData(U(:, 1:36)');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============= Part 6: Dimension Reduction for Faces =================
%  Project images to the eigen space using the top k eigenvectors 
%  If you are applying a machine learning algorithm 
fprintf('\nDimension reduction for face dataset.\n\n');

K = 100;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d ', size(Z));

fprintf('\n\nProgram paused. Press enter to continue.\n');
pause;

%% ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
%  Project images to the eigen space using the top K eigen vectors and 
%  visualize only using those K dimensions
%  Compare to the original input, which is also displayed

fprintf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 100;
X_rec  = recoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;

fprintf('Program paused. Press enter to continue.\n');
pause;


%% === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
%  One useful application of PCA is to use it to visualize high-dimensional
%  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
%  pixel colors of an image. We first visualize this output in 3D, and then
%  apply PCA to obtain a visualization in 2D.

close all; close all; clc

% Re-load the image from the previous exercise and run K-Means on it
% For this to work, you need to complete the K-Means assignment first
A = double(imread('bird_small.png'));

% If imread does not work for you, you can try instead
%   load ('bird_small.mat');

A = A / 255;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 16; 
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

%  Sample 1000 random indexes (since working with all the data is
%  too expensive. If you have a fast computer, you may increase this.
sel = floor(rand(1000, 1) * size(X, 1)) + 1;

%  Setup Color Palette
palette = hsv(K);
colors = palette(idx(sel), :);

%  Visualize the data and centroid memberships in 3D
figure;
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
% Use PCA to project this cloud to 2D for visualization

% Subtract the mean to use PCA
[X_norm, mu, sigma] = featureNormalize(X);

% PCA and project the data to 2D
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);

% Plot in 2D
figure;
plotDataPoints(Z(sel, :), idx(sel), K);
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
fprintf('Program paused. Press enter to continue.\n');
pause;
