
caffe_root = '../../';
addpath(genpath([caffe_root,'matlab/']));
%im = imread('test2.jpg');
im = zeros([50, 150, 3], 'uint8');
use_gpu = 1;

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
net_model = [model_dir 'fcn2_deploy.prototxt'];
net_weights = [model_dir 'models/fcn2/fcn_iter_5000.caffemodel'];
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
if 1
    mean_data = [104.00698793,116.66876762,122.67891434];
    %mean_data = [6, 6, 6];
else
    mean_data = [0, 0, 0];
end
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
mean_data = repmat(mean_data, [size(im_data, 1) * size(im_data, 2), 1]);
mean_data = reshape(mean_data, size(im_data));
im_data = im_data - mean_data;

tic;
scores = net.forward({im_data});
toc;

score_map = scores{1}(:,:,2)'; %score_map(score_map < 0.5) = 0;
imshow(score_map);
[b, b1] = bwlabeln(logical(score_map), 4);
segs_id = unique(b);

nignore = 0;
for i = 1:length(segs_id)
   ind = find(b == segs_id(i));
   if(length(ind) < 0)
      b1 = b1 - 1;
      b(ind) = 0;
      score_map(ind) = 0;
   end
end
yx = zeros([b1 - 1, 2]);
segs_id = unique(b);
figure(1); 
subplot(3,3, 1); imshow(im);
for i = 1:length(segs_id)
   z = false([siz(1), siz(2)]);
   [y, x] = find(b == segs_id(i));
   ind = find(b == segs_id(i));
   if(length(y) < numel(score_map) / 200)
      nignore = nignore + 1; 
      b(ind) = 0;
      score_map(ind) = 0;
   end
   z(ind) = true;
   subplot(3,3,i + 1);imshow(z);title(num2str(segs_id(i)));
   y1 = round(mean(y));
   x1 = round(mean(x));
   yx(i - nignore,:) = [y1, x1];
end
caffe.reset_all();



