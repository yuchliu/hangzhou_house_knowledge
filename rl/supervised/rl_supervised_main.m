function [net,info] = rl_supervised_main(model, categoryName)

gpuDevice();
map_width = 6;
rl_root = 'rl';
opts.expDir=fullfile(rl_root, 'model', model);
opts.imdbPath=fullfile(opts.expDir, 'imdb.mat');
opts.whitenData=true;
opts.contrastNormalization=true;
opts.train=struct('gpus',1);
opts.networkType='simplenn';
vl_setupnn;

%% Initialize Model and Data
net = rl_init(map_width);
if ~exist(opts.expDir,'dir')
    mkdir(opts.expDir);
end
if ~exist(opts.imdbPath,'file')
    useNegative = false;
    imdb = rl_getImdb(model, categoryName, useNegative);
    %imdb = rl_getImdb_fineTune(1, 3);
    save(opts.imdbPath,'-struct','imdb');
else
    imdb=load(opts.imdbPath) ;
end
opts.expDir=fullfile(opts.expDir, 'result');
mkdir(opts.expDir);

%% Train
[net,info]=rl_cnn_train(net,imdb,getBatch(opts),'expDir',opts.expDir,net.meta.trainOpts,opts.train,'val',find(imdb.images.set==2));
end


%% for getting batch
function fn = getBatch(opts)
bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) getSimpleBatch(bopts,x,y) ;
end


function [images,labels]=getSimpleBatch(opts, imdb, batch)
images = imdb.images.data(:,:,:,batch) ;
labels = reshape(imdb.images.labels(:,batch),[1,1,size(imdb.images.labels,1),numel(batch)]);
if opts.numGpus > 0
  images = gpuArray(images) ;
end
end