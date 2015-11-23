function seg_map_s = test_FCN_s(im, use_gpu)
caffe_root = '../../';
addpath(genpath([caffe_root,'matlab/']));
if nargin < 1
  % For demo purposes we will use the cat image
  %fprintf('using caffe/examples/images/cat.jpg as input image\n');
  im = imread('test1.jpg');
end
if nargin < 2
    use_gpu = 1;
end
% if exist('../+caffe', 'dir')
%   addpath('..');
% else
%   error('Please run this demo from caffe/matlab/demo');
% end

% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

model_dir = './';
%model_dir = '../../examples/finetune/';
net_model = [model_dir 'deploy.prototxt'];
net_weights = [model_dir 'train_iter_2500.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);
siz=size(im);
blobdata = net.blob_vec(net.name2blob_index('data'));
oldshape=blobdata.shape;
newshape=[siz(2),siz(1),oldshape(3),oldshape(4)];
blobdata.reshape(newshape);

mean_data = [104.00698793,116.66876762,122.67891434];
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
%input_data = {prepare_image(im)};
mean_data = repmat(mean_data, [size(im_data, 1) * size(im_data, 2), 1]);
mean_data = reshape(mean_data, size(im_data));
im_data = im_data - mean_data;
tic;
% The net forward function. It takes in a cell array of N-D arrays
% (where N == 4 here) containing data of input blob(s) and outputs a cell
% array containing data from output blob(s)

scores = net.forward({im_data});
toc;

 seg_map_s = scores{1};
%  seg_map=seg_map*(-1);
%  map1=seg_map/(max(max(seg_map)));
%  map1=seg_map(:,:,2)';
 map1=seg_map_s(:,:,2)';
 colormap('hot');
 imagesc(map1);
 colorbar;
 caffe.reset_all();

end

function data = prepare_image(im, blob_size, mean_data)
if nargin < 2
    blob_size = 500;
end
if nargin < 2
    mean_data = [104.00698793,116.66876762,122.67891434];
end


im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [blob_size blob_size], 'bilinear');
mean_data = repmat(mean_data, [size(im_data, 1) * size(im_data, 2), 1]);
mean_data = reshape(mean_data, size(im_data));

im_data = im_data - mean_data;
data = zeros(size(im_data, 1), size(im_data, 2), size(im_data, 3), 1, 'single');
data(:, :, :, 1) = im_data;
end