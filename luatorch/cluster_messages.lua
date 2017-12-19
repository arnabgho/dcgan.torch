require 'image'
require 'nn'
npy4th=require 'npy4th'
json=require 'json'
local optnet = require 'optnet'
m=require 'manifold'
torch.setdefaulttensortype('torch.DoubleTensor')

opt = {
    batchSize = 64,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',  -- random / line / linefull1d / linefull
    name = 'generation1',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,           -- Display image: 0 = false, 1 = true
    nz = 100,
    num_tsne=1000,         -- Number of tsne files to visualize
    nmsg=50,                -- Message Dimension              
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

if opt.gpu~=0 then
   require 'cunn'
   --torch.setdefaulttensortype('torch.CudaTensor')
end
jsons={
'bald.json',
'brown.json',
'black.json',
'blond.json',
'gray.json'
}
G=torch.load(opt.net).G
for k,v in pairs(jsons) do
   local file=io.open('/home1/arnab/dcgan.torch/celebA/'..v)
   local file_contents=file:read("*all")
   local filenames=json.decode(file_contents)
   file:close()   
   local data=torch.Tensor(opt.num_tsne,3,64,64)
   local messages=torch.Tensor(opt.num_tsne,opt.nmsg)
   i=1
   for filename,_ in pairs(filenames) do
      data[i]=image.load('/home1/arnab/dcgan.torch/celebA/img_align_celeba/'..filename)  
      i=i+1
      if i>opt.num_tsne then break end
   end   

   local num_batches=opt.num_tsne/opt.batchSize
   local message_batch 
   data=data:cuda()
   for j=0,num_batches-1 do
      --messages[{ {j*opt.batchSize+1,j*opt.batchSize+opt.batchSize} ,{}  }]=G.netI:forward(data[{{j*opt.batchSize+1,j*opt.batchSize+opt.batchSize} ,{} , {} , {}   }]):reshape(opt.batchSize,opt.nmsg)
      message_batch=G.netI:forward(data[{{j*opt.batchSize+1,j*opt.batchSize+opt.batchSize} ,{} , {} , {}   }]):reshape(opt.batchSize,opt.nmsg)
      messages[{ {j*opt.batchSize+1,j*opt.batchSize+opt.batchSize} ,{}  }]  = message_batch:float()
   end

   messages=messages:double()
   local messages_tsne = m.embedding.tsne(messages,{dim=2,perplexity=30})
   --local messages_lle = m.embedding.lle(messages,{dim=2,neighbors=5})
   npy4th.savenpy("celebA/".. v ..'.npy',messages_tsne)
end 










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
