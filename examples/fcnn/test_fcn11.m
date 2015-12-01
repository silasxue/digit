caffe_root = '../../';
addpath(genpath([caffe_root,'matlab/']));
use_gpu = 1;

% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end
test_imgs_dir = '../../data/figure/';
model_dir = './';
%model_dir = '../../examples/finetune/';
net_model = [model_dir 'fcn11_deploy.prototxt'];
net_weights = [model_dir 'models/fcn11/model_iter_15000.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end
% Initialize a network
net = caffe.Net(net_model, net_weights, phase);
%get data
test_imgs = dir([test_imgs_dir,'*.jpg']);
test_imgs = {test_imgs.name};
for i = 1:length(test_imgs),test_imgs{i} = test_imgs{i}(1:end-4);end
for i = 1:length(test_imgs)
    im = imread([test_imgs_dir,test_imgs{i},'.jpg']);
    siz=size(im);
    blobdata = net.blob_vec(net.name2blob_index('data'));
    oldshape=blobdata.shape;
    newshape=[siz(2),siz(1),oldshape(3),oldshape(4)];
    blobdata.reshape(newshape);
    im_data = im(:, :, [3, 2, 1]);   % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);       % convert from uint8 to single
    scores = net.forward({im_data});
    s = scores{1};
    figure(1);
    subplot(3, 4, 1); imshow(im);
    for j = 1:10
       subplot(3, 4, j + 1);
       imshow(s(:,:,j)');
       title(num2str(j - 1));
    end
    colormap('hot');
end
caffe.reset_all();