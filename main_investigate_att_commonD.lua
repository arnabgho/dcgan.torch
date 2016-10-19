require 'torch'
require 'nn'
require 'optim'
local model_utils = require 'util.model_utils'
opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 0,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment-investigate-att1-commonD',
   noise = 'normal',       -- uniform / normal
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local G={}
G.netG1 = nn.Sequential()
-- input is Z, going into a convolution
G.netG1:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
G.netG1:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
G.netG1:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
G.netG1:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
G.netG1:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
G.netG1:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
G.netG1:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
G.netG1:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
G.netG1:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
G.netG1:add(nn.Tanh())
-- state size: (nc) x 64 x 64

G.netG1:apply(weights_init)


G.netG2 = nn.Sequential()
-- input is Z, going into a convolution
G.netG2:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
G.netG2:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
G.netG2:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
G.netG2:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
G.netG2:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
G.netG2:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
G.netG2:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
G.netG2:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
G.netG2:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
G.netG2:add(nn.Tanh())
-- state size: (nc) x 64 x 64

G.netG2:apply(weights_init)

G.netI = nn.Sequential()

-- input is (nc) x 64 x 64
G.netI:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
G.netI:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
G.netI:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
G.netI:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
G.netI:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
G.netI:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
G.netI:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
G.netI:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
G.netI:add(SpatialConvolution(ndf * 8, 1, 4, 4))
G.netI:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
G.netI:add(nn.View(1):setNumInputDims(3))
-- state size: 1

G.netI:apply(weights_init)

G.netI_clone=G.netI:clone('weight','bias','gradWeight','gradBias')

G.netI_clone:apply(weights_init)

G.softmax_I=nn.SoftMax()
G.MM1=nn.MM()
G.MM2=nn.MM()

--local netD = nn.Sequential()
--
---- input is (nc) x 64 x 64
--netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
--netD:add(nn.LeakyReLU(0.2, true))
---- state size: (ndf) x 32 x 32
--netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
--netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
---- state size: (ndf*2) x 16 x 16
--netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
--netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
---- state size: (ndf*4) x 8 x 8
--netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
--netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
---- state size: (ndf*8) x 4 x 4
--netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
--netD:add(nn.Sigmoid())
---- state size: 1 x 1 x 1
--netD:add(nn.View(1):setNumInputDims(3))
---- state size: 1

local netD=G.netI:clone('weight','bias','gradWeight','gradBias')
netD:apply(weights_init)

local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      --cudnn.convert(netG, cudnn)
      for k,net in pairs(G) do cudnn.convert(net,cudnn)  end
      cudnn.convert(netD, cudnn)
   end
   netD:cuda();           --netG:cuda();          
   criterion:cuda()
   for k,net in pairs(G) do net:cuda() end
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = model_utils.combine_all_parameters(G)

if opt.display then disp = require 'display' end

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

local input_att_softmax=torch.Tensor(2,opt.batchSize)
-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = data:getBatch()
   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input)
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake1 = G.netG1:forward(noise)
   local fake2 = G.netG2:forward(noise)

   local att1 = G.netI:forward(fake1)
   local att2 = G.netI_clone:forward(fake2)
   local att12 = torch.cat(att1,att2,2)   -- size (batch_size , 2)
  
   local att12_softmax=G.softmax_I:forward(att12)
   input_att_softmax=att12_softmax:transpose(1,2)
   att1=att12_softmax:transpose(1,2)[1]
   att2=att12_softmax:transpose(1,2)[2]
   
   fake1:resize(opt.batchSize,nc * 64 * 64,1)
   fake2:resize(opt.batchSize,nc * 64 * 64,1)
   
   att1:resize(opt.batchSize,1,1)
   att2:resize(opt.batchSize,1,1)

   local fake1_att1=G.MM1:forward({fake1,att1})
   local fake2_att2=G.MM2:forward({fake2,att2})


   local fake=fake1_att1+fake2_att2
   fake:resize(opt.batchSize,nc,64,64)
   input:copy(fake)
   label:fill(fake_label)

   local output = netD:forward(input)
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()
   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   -- assuming all the required outputs have been precomputed in fDx
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)

   local df_dg = netD:updateGradInput(input, df_do)

   local df_fake1_att1=G.MM1:backward({G.netG1.output:reshape(opt.batchSize,nc * 64 * 64,1), input_att_softmax[1]:reshape(opt.batchSize,1,1) },df_dg:reshape(opt.batchSize,nc * 64 * 64,1))
   
   local df_fake2_att2=G.MM2:backward({G.netG2.output:reshape(opt.batchSize,nc * 64 * 64,1), input_att_softmax[2]:reshape(opt.batchSize,1,1) },df_dg:reshape(opt.batchSize,nc * 64 * 64,1))

   local req_softmax=torch.cat(  df_fake1_att1[2]:reshape(opt.batchSize)  ,  df_fake2_att2[2]:reshape(opt.batchSize)   ,  2  )
   local df_softmax=G.softmax_I:backward(input_att_softmax:transpose(1,2), req_softmax  ):transpose(1,2)

   local df_dI = G.netI:backward(G.netG1.output:reshape(opt.batchSize,nc,64,64,1),df_softmax[1]:reshape(opt.batchSize))
   local df_dI_clone = G.netI_clone:backward(G.netG2.output:reshape(opt.batchSize,nc,64,64,1),df_softmax[2]:reshape(opt.batchSize))
   
   df_dI:add(df_fake1_att1[1]:reshape(opt.batchSize,nc,64,64,1))
   df_dI_clone:add(df_fake2_att2[1]:reshape(opt.batchSize,nc,64,64,1))

   G.netG1:backward(noise,df_dI)
   G.netG2:backward(noise,df_dI_clone)

   return errG, gradParametersG
end

local function generate_samples(noise)
   local fake1 = G.netG1:forward(noise)
   local fake2 = G.netG2:forward(noise)

   local att1 = G.netI:forward(fake1)
   local att2 = G.netI_clone:forward(fake2)
   local att12 = torch.cat(att1,att2,2)   -- size (batch_size , 2)
  
   local att12_softmax=G.softmax_I:forward(att12)
   att1=att12_softmax:transpose(1,2)[1]
   att2=att12_softmax:transpose(1,2)[2]
   
   fake1:resize(opt.batchSize,nc * 64 * 64,1)
   fake2:resize(opt.batchSize,nc * 64 * 64,1)
   
   att1:resize(opt.batchSize,1,1)
   att2:resize(opt.batchSize,1,1)

   local fake1_att1=G.MM1:forward({fake1,att1})
   local fake2_att2=G.MM2:forward({fake2,att2})


   local fake=fake1_att1+fake2_att2
   fake:resize(opt.batchSize,nc,64,64)
   return fake
end


-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local fake = generate_samples(noise_vis)
          local real = data:getBatch()
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints_investigate_att')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   torch.save('checkpoints_investigate_att/' .. opt.name .. '_' .. epoch .. '_net_G.t7', {G.netG1:clearState(),G.netG2:clearState(),G.netI:clearState() } )
   torch.save('checkpoints_investigate_att/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   --parametersG, gradParametersG = netG:getParameters()
   parametersG, gradParametersG = model_utils.combine_all_parameters(G)
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
