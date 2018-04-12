function [net,info]=learn_icnn(model,Name_batch)
load(['./mat/',Name_batch,'/conf.mat'],'conf');
conf.data.Name_batch=Name_batch;
opts.dataDir=conf.data.imgdir;
opts.expDir=fullfile(conf.output.dir,conf.data.Name_batch);
opts.imdbPath=fullfile(opts.expDir,'imdb.mat');
opts.whitenData=true;
opts.contrastNormalization=true;
opts.networkType='simplenn';
opts.fineTunePath=fullfile('rl', 'fineTune data', model);
try
    gpuDevice();
    opts.train=struct('gpus',1);
catch
    error('Errors here: GPU invalid.\n')
end

%% Prepare model
labelNum=1;
net=network_init(labelNum,model,'networkType',opts.networkType);


%% Prepare data
if exist(opts.imdbPath,'file')
  imdb=load(opts.imdbPath) ;
else
  IsTrain=true;
  if(strcmp(Name_batch,'cub200'))
      imdb=getImdb_cub200(conf.data.Name_batch,conf,net.meta,IsTrain);
  else
      imdb=getImdb(conf.data.Name_batch,conf,net.meta,IsTrain);
  end
  mkdir(opts.expDir);
  save(opts.imdbPath,'-struct','imdb');
end

net.meta.classes.name=imdb.meta.classes(:)';
%net = load_rl_net(net, 'rl/model/vgg-s/bird/result/net-epoch-3.mat');


%% Train
rl_net_name = 'net-epoch-3.mat';
rl_model_path = fullfile('rl/model', model, 'result', rl_net_name);
%rl_model_path = 'rl/model/vgg-s/bird/result/e7s1.mat';
[net,info]=our_cnn_train(net,imdb,getBatch(opts),...
                        'expDir',opts.expDir,net.meta.trainOpts,opts.train,'val',find(imdb.images.set==2),...
                        'load_rl_net', load_rl_net(rl_model_path),...
                        'fineTunePath', opts.fineTunePath);
end


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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fn = load_rl_net(model_dir)
fn = @(net) load_rl_net_simple(net, model_dir); 
end


function net = load_rl_net_simple( net, model_dir )
%INITMETALEARNPARAM Summary of this function goes here
%   Detailed explanation goes here
    for i=numel(net.layers):-1:1
       if strcmp(net.layers{i}.type, 'conv_mask')
           net.layers{end}.rl_layer_id = i;
           model = load(model_dir, 'net');
           net.layers{i}.rl_net = model.net;
           net.layers{i}.rl_net.layers(end) = [];
           fprintf('Add RL net at layer %d.\n', i);
           break;
       end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%