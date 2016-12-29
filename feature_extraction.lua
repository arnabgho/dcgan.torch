require 'image'
require 'nn'
npy4th=require 'npy4th'
json=require 'json'
local optnet = require 'optnet'
m=require 'manifold'
require 'hdf5'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 64,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    netD = '',             -- path to the discriminator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',  -- random / line / linefull1d / linefull
    name = 'generation1',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,           -- Display image: 0 = false, 1 = true
    nz = 100,
    num_tsne=1024,         -- Number of tsne files to visualize
    nmsg=50,                -- Message Dimension              
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
local cifar_train=torch.load('cifar/cifar10-train.t7')
local cifar_test =torch.load('cifar/cifar10-test.t7')

local num_batches_train =cifar_train.data:size(1)/opt.batchSize
local num_batches_test = cifar_test.data:size(1)/opt.batchSize

--local features_train = torch.Tensor(num_batches_train*opt.batchSize, 2*512*4*4)
local features_test = torch.Tensor(num_batches_test*opt.batchSize, 2*512*4*4)
local image_batch=torch.Tensor(opt.batchSize,3,64,64)

if opt.gpu~=0 then
    require 'cunn'
    --cifar_train.data=cifar_train.data:cuda();
--    cifar_test.data=cifar_test.data:cuda();
--    features_train=features_train:cuda();
    features_test=features_test:cuda()
    image_batch=image_batch:cuda()
end

assert(net ~= '', 'provide a generator model')

assert(netD ~= '', 'provide a generator model')
local G=torch.load(opt.net).G
local D=torch.load(opt.netD)


--for i=0,num_batches_train-1 do
--   for j=1,opt.batchSize do
--	image_batch[j]=image.scale(cifar_train.data[i*opt.batchSize+j],64,64)
--   end 
--   D:forward( image_batch )
--   local D_features=D.modules[11].output:reshape(opt.batchSize,512*4*4)
--   G.netI:forward( image_batch )
--   local M_features=G.netI.modules[11].output:reshape(opt.batchSize,512*4*4)
--   features_train[ {{i*opt.batchSize+1 , (i+1)*opt.batchSize   }}  ]=torch.cat( {D_features , M_features} ,2  )
--end
for i=0,num_batches_test-1 do
   for j=1,opt.batchSize do
	image_batch[j]=image.scale(cifar_test.data[i*opt.batchSize+j],64,64)
   end 
   D:forward( image_batch )
   local D_features=D.modules[11].output:reshape(opt.batchSize,512*4*4)
   G.netI:forward( image_batch )
   local M_features=G.netI.modules[11].output:reshape(opt.batchSize,512*4*4)
   features_test[ {{i*opt.batchSize+1 , (i+1)*opt.batchSize   }}  ]=torch.cat( {D_features , M_features} ,2  )
end


--for i=0,num_batches_test-1 do
--   D:forward( cifar_test.data[{{i*opt.batchSize+1,(i+1)*opt.batchSize }}] )
--   local D_features=D.modules[11].output:reshape(opt.batchSize,512*4*4)
--   G.net_I:forward(   cifar_test.data[{{i*opt.batchSize+1,(i+1)*opt.batchSize }}] )
--   local M_features=G.net_I.modules[11].output:reshape(opt.batchSize,512*4*4)
--   features_test[ {{i*opt.batchSize+1 , (i+1)*opt.batchSize   }}  ]=torch.cat( {D_features , M_features} ,2  )
--end
--features_train=features_train:float()
features_test = features_test:float()

--local train_file=hdf5.open( "imgnet_repTrain.h5" ,'w' )
--train_file:write('features',features_train)
--train_file:write('labels',  cifar_train.label[ {{ 1 , num_batches_train*opt.batchSize  }}  ] )
local test_file=hdf5.open( "imgnet_repTest.h5" ,'w' )
test_file:write('features',features_test)
test_file:write('labels',  cifar_test.label[ {{ 1 , num_batches_test*opt.batchSize  }}  ] )


--npy4th.savenpy( 'repTrain.npy' , features_train  )
--npy4th.savenpy( 'classTrain.npy', cifar_train.label[ {{ 1 , num_batches_train*opt.batchSize  }}  ])
--npy4th.savenpy( 'repTest.npy' , features_test  )
--npy4th.savenpy( 'classTest.npy', cifar_test.label[ {{ 1 , num_batches_test*opt.batchSize  }}  ])





--for k,v in pairs(jsons) do
--   local file=io.open('celebA/'..k)
--   local file_contents=file:read("*all")
--   local filenames=json.decode(file_contents)
--   file:close()   
--   local data=torch.Tensor(opt.num_tsne,3,64,64)
--   local messages=torch.Tensor(opt.num_tsne,opt.nmsg)
--   local i=1
--   for filename,_ in pairs(filenames) do
--      data[i]=image.load('celebA/img_align_celebA/'..filename)  
--      i=i+1
--      if i>opt.num_tsne then break end
--   end   
--
--   local num_batches=opt.num_tsne/opt.batchSize
-- 
--   for j=0,num_batches-1 do
--      messages[{{j*opt.batchSize+1,j*opt.batchSize+opt.batchSize}}]=G.net_I:forward(data[{{j*opt.batchSize+1,j*opt.batchSize+opt.batchSize}}]):reshape(opt.batchSize,opt.nmsg)
--   end
--   local messages_tsne = m.embeddin.tsne(messages,{dim=2,perplexity=30})
--   npy4th.savenpy("celebA/"..k..'.npy',messages_tsne)
--end 










--
--
--noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
--net = torch.load(opt.net)
--
---- for older models, there was nn.View on the top
---- which is unnecessary, and hinders convolutional generations.
--if torch.type(net:get(1)) == 'nn.View' then
--    net:remove(1)
--end
--
--print(net)
--
--if opt.noisetype == 'uniform' then
--    noise:uniform(-1, 1)
--elseif opt.noisetype == 'normal' then
--    noise:normal(0, 1)
--end
--
--noiseL = torch.FloatTensor(opt.nz):uniform(-1, 1)
--noiseR = torch.FloatTensor(opt.nz):uniform(-1, 1)
--if opt.noisemode == 'line' then
--   -- do a linear interpolation in Z space between point A and point B
--   -- each sample in the mini-batch is a point on the line
--    line  = torch.linspace(0, 1, opt.batchSize)
--    for i = 1, opt.batchSize do
--        noise:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
--    end
--elseif opt.noisemode == 'linefull1d' then
--   -- do a linear interpolation in Z space between point A and point B
--   -- however, generate the samples convolutionally, so a giant image is produced
--    assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
--    noise = noise:narrow(3, 1, 1):clone()
--    line  = torch.linspace(0, 1, opt.imsize)
--    for i = 1, opt.imsize do
--        noise:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
--    end
--elseif opt.noisemode == 'linefull' then
--   -- just like linefull1d above, but try to do it in 2D
--    assert(opt.batchSize == 1, 'for linefull mode, give batchSize(1) and imsize > 1')
--    line  = torch.linspace(0, 1, opt.imsize)
--    for i = 1, opt.imsize do
--        noise:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
--    end
--end
--
--local sample_input = torch.randn(2,100,1,1)
--if opt.gpu > 0 then
--    require 'cunn'
--    require 'cudnn'
--    net:cuda()
--    cudnn.convert(net, cudnn)
--    noise = noise:cuda()
--    sample_input = sample_input:cuda()
--else
--   sample_input = sample_input:float()
--   net:float()
--end
--
---- a function to setup double-buffering across the network.
---- this drastically reduces the memory needed to generate samples
--optnet.optimizeMemory(net, sample_input)
--
--local images = net:forward(noise)
--print('Images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
--images:add(1):mul(0.5)
--print('Min, Max, Mean, Stdv', images:min(), images:max(), images:mean(), images:std())
--image.save(opt.name .. '.png', image.toDisplayTensor(images))
--print('Saved image to: ', opt.name .. '.png')
--
--if opt.display then
--    disp = require 'display'
--    disp.image(images)
--    print('Displayed image')
--end
