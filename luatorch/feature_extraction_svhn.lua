require 'image'
require 'nn'
npy4th=require 'npy4th'
json=require 'json'
local optnet = require 'optnet'
m=require 'manifold'
require 'hdf5'
torch.setdefaulttensortype('torch.FloatTensor')
require 'rnn'
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
    mode="train",          -- mode           
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
local train=torch.load('svhn/svhn_train.t7')
local test =torch.load('svhn/svhn_test.t7')

local num_batches_train = train.data:size(1)/opt.batchSize
local num_batches_test = test.data:size(1)/opt.batchSize

if opt.mode=="train" then
	features_train = torch.Tensor(num_batches_train*opt.batchSize, 2*512*4*4)
end
if opt.mode=="test" then
	features_test = torch.Tensor(num_batches_test*opt.batchSize, 2*512*4*4)
end
local image_batch=torch.Tensor(opt.batchSize,3,64,64)

if opt.gpu~=0 then
    require 'cunn'
    --train.data=train.data:cuda();
--    test.data=test.data:cuda();
--    features_train=features_train:cuda();
--    features_test=features_test:cuda()
    image_batch=image_batch:cuda()
end

assert(net ~= '', 'provide a generator model')

assert(netD ~= '', 'provide a generator model')
local G=torch.load(opt.net).G
local D=torch.load(opt.netD)

if opt.mode=="train" then
for i=0,math.floor(num_batches_train)-1 do
   for j=1,opt.batchSize do
	image_batch[j]=image.scale(train.data[i*opt.batchSize+j],64,64)
   end 
   D:forward( image_batch )
   local D_features=D.modules[11].output:reshape(opt.batchSize,512*4*4):float()
   G.netI:forward( image_batch )
   local M_features=G.netI.modules[11].output:reshape(opt.batchSize,512*4*4):float()
   features_train[ {{i*opt.batchSize+1 , (i+1)*opt.batchSize   }}  ]=torch.cat( {D_features , M_features} ,2  )
end

local train_file=hdf5.open( "svhn_repTrain.h5" ,'w' )
train_file:write('features',features_train)
train_file:write('labels',  train.label[ {{ 1 , num_batches_train*opt.batchSize  }}  ] )
end

if opt.mode=="test" then
for i=0,num_batches_test-1 do
   for j=1,opt.batchSize do
	image_batch[j]=image.scale(test.data[i*opt.batchSize+j],64,64)
   end 
   D:forward( image_batch )
   local D_features=D.modules[11].output:reshape(opt.batchSize,512*4*4):float()
   G.netI:forward( image_batch )
   local M_features=G.netI.modules[11].output:reshape(opt.batchSize,512*4*4):float()
   features_test[ {{i*opt.batchSize+1 , (i+1)*opt.batchSize   }}  ]=torch.cat( {D_features , M_features} ,2  )
end



local test_file=hdf5.open( "svhn_repTest.h5" ,'w' )
test_file:write('features',features_test)
test_file:write('labels',  test.label[ {{ 1 , num_batches_test*opt.batchSize  }}  ] )

end

--npy4th.savenpy( 'repTrain.npy' , features_train  )
--npy4th.savenpy( 'classTrain.npy', train.label[ {{ 1 , num_batches_train*opt.batchSize  }}  ])
--npy4th.savenpy( 'repTest.npy' , features_test  )
--npy4th.savenpy( 'classTest.npy', test.label[ {{ 1 , num_batches_test*opt.batchSize  }}  ])






