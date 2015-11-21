caffe_root = '../../';
data_root = [caffe_root, '/data/digit/'];
test_imgs = load([data_root, 'test.mat']);
test_imgs = test_imgs.pair;

addpath(genpath([caffe_root, 'matlab']));
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
model_dir = './';
net_model = [model_dir 'lenet.prototxt'];
net_weights = [model_dir 'digit_iter_5000.caffemodel'];
phase = 'test';
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end
net = caffe.Net(net_model, net_weights, phase);

idx = 8678;

im = imread([data_root,test_imgs{idx}]);
im_data = im(:, :, [3, 2, 1]);
im_data = permute(im_data, [2, 1, 3]);
im_data = single(im_data);
im_data = im_data(:,:,1);
input_data = {single(im_data)};

probabilities = net.forward(input_data);probabilities = probabilities{:};
imshow(im);
[~, rst] = max(probabilities);
probabilities
disp(['----> ',num2str(rst - 1),' <-----']);